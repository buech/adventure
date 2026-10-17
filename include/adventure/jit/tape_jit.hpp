/*
 * Copyright (c) 2025-2026 Adam Buechner
 *
 * Licensed under the Apache License 2.0, or, at your option, the MIT License.
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#pragma once

#ifdef ADVENTURE_ENABLE_LLVM_JIT

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/TargetSelect.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <type_traits>

#include "adventure/config.hpp"
#include "adventure/ops.hpp"

namespace adventure {

// Forward declaration
template <typename T>
class Tape;

namespace jit {

template <typename T>
using PrimalKernel = void (*)(T * /*primals*/);

template <typename T>
using BackwardKernel = void (*)(const T * /*primals*/, T * /*adj*/);

/// JIT compilation and execution engine for Tape operations using LLVM ORC JIT.
template <typename T>
class TapeJITEngine {
 public:
  TapeJITEngine() = default;

  TapeJITEngine(const TapeJITEngine &) = delete;
  TapeJITEngine &operator=(const TapeJITEngine &) = delete;

  TapeJITEngine(TapeJITEngine &&) noexcept = default;
  TapeJITEngine &operator=(TapeJITEngine &&) noexcept = default;

  /// Ensure native target is initialized once.
  static void initialize_llvm() {
    static bool initialized = []() {
      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmPrinter();
      return true;
    }();
    (void)initialized;
  }

  /// Check if the JIT has compiled kernels available.
  [[nodiscard]] bool is_compiled() const noexcept {
    return primal_fn_ != nullptr && backward_fn_ != nullptr;
  }

  /// Invalidate and reset any previously compiled kernels.
  void reset() noexcept {
    primal_fn_ = nullptr;
    backward_fn_ = nullptr;
    jit_.reset();
  }

  /// Compile the given tape's forward and backward graph into native machine
  /// code.
  bool compile(const Tape<T> &tape, bool optimize = true) {
    if (tape.primals.empty()) {
      reset();
      return false;
    }

    initialize_llvm();

    auto jit_builder = llvm::orc::LLJITBuilder();
    auto jit_or_err = jit_builder.create();
    if (!jit_or_err) {
      llvm::consumeError(jit_or_err.takeError());
      return false;
    }
    jit_ = std::move(*jit_or_err);

    auto ctx = std::make_unique<llvm::LLVMContext>();
    auto mod = std::make_unique<llvm::Module>("adventure_tape_jit", *ctx);

    emit_primal_function(*mod, *ctx, tape);
    emit_backward_function(*mod, *ctx, tape);

    if (optimize) {
      optimize_module(*mod);
    }

    llvm::orc::ThreadSafeModule tsm(std::move(mod), std::move(ctx));
    if (auto err = jit_->addIRModule(std::move(tsm))) {
      llvm::consumeError(std::move(err));
      reset();
      return false;
    }

    auto primal_sym = jit_->lookup("tape_primal_kernel");
    auto backward_sym = jit_->lookup("tape_backward_kernel");

    if (!primal_sym || !backward_sym) {
      if (!primal_sym) llvm::consumeError(primal_sym.takeError());
      if (!backward_sym) llvm::consumeError(backward_sym.takeError());
      reset();
      return false;
    }

    primal_fn_ = primal_sym->template toPtr<PrimalKernel<T>>();
    backward_fn_ = backward_sym->template toPtr<BackwardKernel<T>>();
    return true;
  }

  /// Execute the compiled primal kernel.
  void run_primal(T *primals) const noexcept {
    assert(primal_fn_ != nullptr && "Tape JIT primal kernel not compiled");
    primal_fn_(primals);
  }

  /// Execute the compiled backward kernel.
  void run_backward(const T *primals, T *adj) const noexcept {
    assert(backward_fn_ != nullptr && "Tape JIT backward kernel not compiled");
    backward_fn_(primals, adj);
  }

 private:
  std::unique_ptr<llvm::orc::LLJIT> jit_{nullptr};
  PrimalKernel<T> primal_fn_{nullptr};
  BackwardKernel<T> backward_fn_{nullptr};

  static llvm::Type *get_scalar_type(llvm::LLVMContext &ctx) {
    if constexpr (std::is_same_v<T, float>) {
      return llvm::Type::getFloatTy(ctx);
    } else {
      static_assert(std::is_same_v<T, double>,
                    "Only float and double are supported for Tape JIT");
      return llvm::Type::getDoubleTy(ctx);
    }
  }

  static void optimize_module(llvm::Module &mod) {
    llvm::PassBuilder pb;
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;

    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    llvm::ModulePassManager mpm =
        pb.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
    mpm.run(mod, mam);
  }

  static llvm::FunctionCallee get_math_function(llvm::Module &mod,
                                                llvm::LLVMContext &ctx,
                                                llvm::Type *scalarTy,
                                                const char *doubleName,
                                                const char *floatName) {
    const char *fnName = std::is_same_v<T, float> ? floatName : doubleName;
    auto fnTy = llvm::FunctionType::get(scalarTy, {scalarTy}, false);
    return mod.getOrInsertFunction(fnName, fnTy);
  }

  static void emit_primal_function(llvm::Module &mod, llvm::LLVMContext &ctx,
                                   const Tape<T> &tape) {
    llvm::Type *scalarTy = get_scalar_type(ctx);
    llvm::Type *ptrTy = llvm::PointerType::getUnqual(ctx);

    llvm::FunctionType *fnTy =
        llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), {ptrTy}, false);
    llvm::Function *fn = llvm::Function::Create(
        fnTy, llvm::Function::ExternalLinkage, "tape_primal_kernel", mod);

    fn->addParamAttr(1, llvm::Attribute::NoAlias);

    llvm::BasicBlock *bb = llvm::BasicBlock::Create(ctx, "entry", fn);
    llvm::IRBuilder<> builder(bb);
    llvm::Value *primalsBase = fn->getArg(0);

    // Helper to load primals[idx]
    auto loadPrimal = [&](std::size_t idx) -> llvm::Value * {
      llvm::Value *ptr =
          builder.CreateGEP(scalarTy, primalsBase, builder.getInt64(idx));
      return builder.CreateLoad(scalarTy, ptr);
    };

    // Helper to store primals[idx]
    auto storePrimal = [&](std::size_t idx, llvm::Value *val) {
      llvm::Value *ptr =
          builder.CreateGEP(scalarTy, primalsBase, builder.getInt64(idx));
      builder.CreateStore(val, ptr);
    };

    std::size_t arg_idx = 0;

    for (std::size_t i = 0; i < tape.ops.size(); ++i) {
      const OpCode op = tape.ops[i];
      llvm::Value *res = nullptr;

      switch (op) {
        case OpCode::CONSTANT:
        case OpCode::LEAF:
          break;

        case OpCode::COPY: {
          index_t p = tape.operands[arg_idx++];
          res = loadPrimal(p);
          break;
        }

        case OpCode::ADD: {
          index_t lhs = tape.operands[arg_idx++];
          index_t rhs = tape.operands[arg_idx++];
          res = builder.CreateFAdd(loadPrimal(lhs), loadPrimal(rhs));
          break;
        }

        case OpCode::SUB: {
          index_t lhs = tape.operands[arg_idx++];
          index_t rhs = tape.operands[arg_idx++];
          res = builder.CreateFSub(loadPrimal(lhs), loadPrimal(rhs));
          break;
        }

        case OpCode::MUL: {
          index_t lhs = tape.operands[arg_idx++];
          index_t rhs = tape.operands[arg_idx++];
          res = builder.CreateFMul(loadPrimal(lhs), loadPrimal(rhs));
          break;
        }

        case OpCode::DIV: {
          index_t lhs = tape.operands[arg_idx++];
          index_t rhs = tape.operands[arg_idx++];
          res = builder.CreateFDiv(loadPrimal(lhs), loadPrimal(rhs));
          break;
        }

        case OpCode::MIN: {
          index_t lhs = tape.operands[arg_idx++];
          index_t rhs = tape.operands[arg_idx++];
          llvm::Function *minFn = llvm::Intrinsic::getOrInsertDeclaration(
              &mod, llvm::Intrinsic::minnum, {scalarTy});
          res = builder.CreateCall(minFn, {loadPrimal(lhs), loadPrimal(rhs)});
          break;
        }

        case OpCode::MAX: {
          index_t lhs = tape.operands[arg_idx++];
          index_t rhs = tape.operands[arg_idx++];
          llvm::Function *maxFn = llvm::Intrinsic::getOrInsertDeclaration(
              &mod, llvm::Intrinsic::maxnum, {scalarTy});
          res = builder.CreateCall(maxFn, {loadPrimal(lhs), loadPrimal(rhs)});
          break;
        }

        case OpCode::SIN: {
          index_t p = tape.operands[arg_idx++];
          llvm::Function *sinFn = llvm::Intrinsic::getOrInsertDeclaration(
              &mod, llvm::Intrinsic::sin, {scalarTy});
          res = builder.CreateCall(sinFn, {loadPrimal(p)});
          break;
        }

        case OpCode::COS: {
          index_t p = tape.operands[arg_idx++];
          llvm::Function *cosFn = llvm::Intrinsic::getOrInsertDeclaration(
              &mod, llvm::Intrinsic::cos, {scalarTy});
          res = builder.CreateCall(cosFn, {loadPrimal(p)});
          break;
        }

        case OpCode::EXP: {
          index_t p = tape.operands[arg_idx++];
          llvm::Function *expFn = llvm::Intrinsic::getOrInsertDeclaration(
              &mod, llvm::Intrinsic::exp, {scalarTy});
          res = builder.CreateCall(expFn, {loadPrimal(p)});
          break;
        }

        case OpCode::LOG: {
          index_t p = tape.operands[arg_idx++];
          llvm::Function *logFn = llvm::Intrinsic::getOrInsertDeclaration(
              &mod, llvm::Intrinsic::log, {scalarTy});
          res = builder.CreateCall(logFn, {loadPrimal(p)});
          break;
        }

        case OpCode::LOG10: {
          index_t p = tape.operands[arg_idx++];
          llvm::Function *log10Fn = llvm::Intrinsic::getOrInsertDeclaration(
              &mod, llvm::Intrinsic::log10, {scalarTy});
          res = builder.CreateCall(log10Fn, {loadPrimal(p)});
          break;
        }

        case OpCode::TANH: {
          index_t p = tape.operands[arg_idx++];
          auto tanhFn = get_math_function(mod, ctx, scalarTy, "tanh", "tanhf");
          res = builder.CreateCall(tanhFn, {loadPrimal(p)});
          break;
        }

        case OpCode::TAN: {
          index_t p = tape.operands[arg_idx++];
          auto tanFn = get_math_function(mod, ctx, scalarTy, "tan", "tanf");
          res = builder.CreateCall(tanFn, {loadPrimal(p)});
          break;
        }

        case OpCode::ATAN: {
          index_t p = tape.operands[arg_idx++];
          auto atanFn = get_math_function(mod, ctx, scalarTy, "atan", "atanf");
          res = builder.CreateCall(atanFn, {loadPrimal(p)});
          break;
        }

        case OpCode::ABS: {
          index_t p = tape.operands[arg_idx++];
          llvm::Function *fabsFn = llvm::Intrinsic::getOrInsertDeclaration(
              &mod, llvm::Intrinsic::fabs, {scalarTy});
          res = builder.CreateCall(fabsFn, {loadPrimal(p)});
          break;
        }

        case OpCode::SQRT: {
          index_t p = tape.operands[arg_idx++];
          llvm::Function *sqrtFn = llvm::Intrinsic::getOrInsertDeclaration(
              &mod, llvm::Intrinsic::sqrt, {scalarTy});
          res = builder.CreateCall(sqrtFn, {loadPrimal(p)});
          break;
        }

        case OpCode::QBRT: {
          index_t p = tape.operands[arg_idx++];
          auto cbrtFn = get_math_function(mod, ctx, scalarTy, "cbrt", "cbrtf");
          res = builder.CreateCall(cbrtFn, {loadPrimal(p)});
          break;
        }

        case OpCode::NEG: {
          index_t p = tape.operands[arg_idx++];
          res = builder.CreateFNeg(loadPrimal(p));
          break;
        }

        case OpCode::POW: {
          index_t lhs = tape.operands[arg_idx++];
          index_t rhs = tape.operands[arg_idx++];
          llvm::Function *powFn = llvm::Intrinsic::getOrInsertDeclaration(
              &mod, llvm::Intrinsic::pow, {scalarTy});
          res = builder.CreateCall(powFn, {loadPrimal(lhs), loadPrimal(rhs)});
          break;
        }
      }

      if (res != nullptr) {
        storePrimal(i, res);
      }
    }

    builder.CreateRetVoid();
  }

  static void emit_backward_function(llvm::Module &mod, llvm::LLVMContext &ctx,
                                     const Tape<T> &tape) {
    llvm::Type *scalarTy = get_scalar_type(ctx);
    llvm::Type *ptrTy = llvm::PointerType::getUnqual(ctx);

    llvm::FunctionType *fnTy = llvm::FunctionType::get(
        llvm::Type::getVoidTy(ctx), {ptrTy, ptrTy}, false);
    llvm::Function *fn = llvm::Function::Create(
        fnTy, llvm::Function::ExternalLinkage, "tape_backward_kernel", mod);

    fn->addParamAttr(0, llvm::Attribute::NoAlias);
    fn->addParamAttr(0, llvm::Attribute::ReadOnly);

    fn->addParamAttr(1, llvm::Attribute::NoAlias);

    llvm::BasicBlock *bb = llvm::BasicBlock::Create(ctx, "entry", fn);
    llvm::IRBuilder<> builder(bb);
    llvm::Value *primalsBase = fn->getArg(0);
    llvm::Value *adjBase = fn->getArg(1);

    auto loadPrimal = [&](std::size_t idx) -> llvm::Value * {
      llvm::Value *ptr =
          builder.CreateGEP(scalarTy, primalsBase, builder.getInt64(idx));
      return builder.CreateLoad(scalarTy, ptr);
    };

    auto loadAdj = [&](std::size_t idx) -> llvm::Value * {
      llvm::Value *ptr =
          builder.CreateGEP(scalarTy, adjBase, builder.getInt64(idx));
      return builder.CreateLoad(scalarTy, ptr);
    };

    auto addAdj = [&](std::size_t idx, llvm::Value *delta) {
      llvm::Value *ptr =
          builder.CreateGEP(scalarTy, adjBase, builder.getInt64(idx));
      llvm::Value *cur = builder.CreateLoad(scalarTy, ptr);
      llvm::Value *updated = builder.CreateFAdd(cur, delta);
      builder.CreateStore(updated, ptr);
    };

    std::size_t arg_idx = tape.operands.size();

    for (std::size_t i = tape.primals.size(); i-- > 0;) {
      const OpCode op = tape.ops[i];
      if (op == OpCode::CONSTANT || op == OpCode::LEAF) {
        continue;
      }

      llvm::Value *seed = loadAdj(i);

      switch (op) {
        case OpCode::CONSTANT:
        case OpCode::LEAF:
          break;

        case OpCode::COPY: {
          const index_t p = tape.operands[--arg_idx];
          addAdj(p, seed);
          break;
        }

        case OpCode::SIN: {
          const index_t p = tape.operands[--arg_idx];
          llvm::Function *cosFn = llvm::Intrinsic::getOrInsertDeclaration(
              &mod, llvm::Intrinsic::cos, {scalarTy});
          llvm::Value *cosVal = builder.CreateCall(cosFn, {loadPrimal(p)});
          addAdj(p, builder.CreateFMul(cosVal, seed));
          break;
        }

        case OpCode::COS: {
          const index_t p = tape.operands[--arg_idx];
          llvm::Function *sinFn = llvm::Intrinsic::getOrInsertDeclaration(
              &mod, llvm::Intrinsic::sin, {scalarTy});
          llvm::Value *sinVal = builder.CreateCall(sinFn, {loadPrimal(p)});
          llvm::Value *negSin = builder.CreateFNeg(sinVal);
          addAdj(p, builder.CreateFMul(negSin, seed));
          break;
        }

        case OpCode::EXP: {
          const index_t p = tape.operands[--arg_idx];
          llvm::Function *expFn = llvm::Intrinsic::getOrInsertDeclaration(
              &mod, llvm::Intrinsic::exp, {scalarTy});
          llvm::Value *expVal = builder.CreateCall(expFn, {loadPrimal(p)});
          addAdj(p, builder.CreateFMul(expVal, seed));
          break;
        }

        case OpCode::LOG: {
          const index_t p = tape.operands[--arg_idx];
          addAdj(p, builder.CreateFDiv(seed, loadPrimal(p)));
          break;
        }

        case OpCode::LOG10: {
          const index_t p = tape.operands[--arg_idx];
          const T ln10 = std::log(T(10));
          llvm::Value *ln10Val =
              llvm::ConstantFP::get(scalarTy, static_cast<double>(ln10));
          llvm::Value *denom = builder.CreateFMul(loadPrimal(p), ln10Val);
          addAdj(p, builder.CreateFDiv(seed, denom));
          break;
        }

        case OpCode::TANH: {
          const index_t p = tape.operands[--arg_idx];
          auto tanhFn = get_math_function(mod, ctx, scalarTy, "tanh", "tanhf");
          llvm::Value *t = builder.CreateCall(tanhFn, {loadPrimal(p)});
          llvm::Value *t2 = builder.CreateFMul(t, t);
          llvm::Value *one = llvm::ConstantFP::get(scalarTy, 1.0);
          llvm::Value *diff = builder.CreateFSub(one, t2);
          addAdj(p, builder.CreateFMul(diff, seed));
          break;
        }

        case OpCode::TAN: {
          const index_t p = tape.operands[--arg_idx];
          llvm::Function *cosFn = llvm::Intrinsic::getOrInsertDeclaration(
              &mod, llvm::Intrinsic::cos, {scalarTy});
          llvm::Value *c = builder.CreateCall(cosFn, {loadPrimal(p)});
          llvm::Value *c2 = builder.CreateFMul(c, c);
          addAdj(p, builder.CreateFDiv(seed, c2));
          break;
        }

        case OpCode::ATAN: {
          const index_t p = tape.operands[--arg_idx];
          llvm::Value *x = loadPrimal(p);
          llvm::Value *x2 = builder.CreateFMul(x, x);
          llvm::Value *one = llvm::ConstantFP::get(scalarTy, 1.0);
          llvm::Value *denom = builder.CreateFAdd(one, x2);
          addAdj(p, builder.CreateFDiv(seed, denom));
          break;
        }

        case OpCode::ABS: {
          const index_t p = tape.operands[--arg_idx];
          llvm::Value *a = loadPrimal(p);
          llvm::Value *zero = llvm::ConstantFP::get(scalarTy, 0.0);
          llvm::Value *one = llvm::ConstantFP::get(scalarTy, 1.0);
          llvm::Value *neg_one = llvm::ConstantFP::get(scalarTy, -1.0);

          llvm::Value *gt = builder.CreateFCmpOGT(a, zero);
          llvm::Value *lt = builder.CreateFCmpOLT(a, zero);
          llvm::Value *sel_neg = builder.CreateSelect(lt, neg_one, zero);
          llvm::Value *sign = builder.CreateSelect(gt, one, sel_neg);
          addAdj(p, builder.CreateFMul(sign, seed));
          break;
        }

        case OpCode::SQRT: {
          const index_t p = tape.operands[--arg_idx];
          llvm::Function *sqrtFn = llvm::Intrinsic::getOrInsertDeclaration(
              &mod, llvm::Intrinsic::sqrt, {scalarTy});
          llvm::Value *sqrtVal = builder.CreateCall(sqrtFn, {loadPrimal(p)});
          llvm::Value *half = llvm::ConstantFP::get(scalarTy, 0.5);
          llvm::Value *deriv = builder.CreateFDiv(half, sqrtVal);
          addAdj(p, builder.CreateFMul(deriv, seed));
          break;
        }

        case OpCode::QBRT: {
          const index_t p = tape.operands[--arg_idx];
          auto cbrtFn = get_math_function(mod, ctx, scalarTy, "cbrt", "cbrtf");
          llvm::Value *c = builder.CreateCall(cbrtFn, {loadPrimal(p)});
          llvm::Value *c2 = builder.CreateFMul(c, c);
          llvm::Value *three = llvm::ConstantFP::get(scalarTy, 3.0);
          llvm::Value *denom = builder.CreateFMul(three, c2);
          addAdj(p, builder.CreateFDiv(seed, denom));
          break;
        }

        case OpCode::NEG: {
          const index_t p = tape.operands[--arg_idx];
          addAdj(p, builder.CreateFNeg(seed));
          break;
        }

        case OpCode::ADD: {
          const index_t rhs = tape.operands[--arg_idx];
          const index_t lhs = tape.operands[--arg_idx];
          addAdj(lhs, seed);
          addAdj(rhs, seed);
          break;
        }

        case OpCode::SUB: {
          const index_t rhs = tape.operands[--arg_idx];
          const index_t lhs = tape.operands[--arg_idx];
          addAdj(lhs, seed);
          addAdj(rhs, builder.CreateFNeg(seed));
          break;
        }

        case OpCode::MUL: {
          const index_t rhs = tape.operands[--arg_idx];
          const index_t lhs = tape.operands[--arg_idx];
          addAdj(lhs, builder.CreateFMul(loadPrimal(rhs), seed));
          addAdj(rhs, builder.CreateFMul(loadPrimal(lhs), seed));
          break;
        }

        case OpCode::DIV: {
          const index_t rhs = tape.operands[--arg_idx];
          const index_t lhs = tape.operands[--arg_idx];
          llvm::Value *one = llvm::ConstantFP::get(scalarTy, 1.0);
          llvm::Value *inv_rhs = builder.CreateFDiv(one, loadPrimal(rhs));
          addAdj(lhs, builder.CreateFMul(inv_rhs, seed));
          llvm::Value *prim_i = loadPrimal(i);
          llvm::Value *adj_rhs_term =
              builder.CreateFMul(builder.CreateFMul(prim_i, inv_rhs), seed);
          addAdj(rhs, builder.CreateFNeg(adj_rhs_term));
          break;
        }

        case OpCode::MIN: {
          const index_t rhs = tape.operands[--arg_idx];
          const index_t lhs = tape.operands[--arg_idx];
          llvm::Value *pLhs = loadPrimal(lhs);
          llvm::Value *pRhs = loadPrimal(rhs);
          llvm::Value *zero = llvm::ConstantFP::get(scalarTy, 0.0);

          llvm::Value *cmpL = builder.CreateFCmpOLT(pLhs, pRhs);
          llvm::Value *deltaL = builder.CreateSelect(cmpL, seed, zero);
          addAdj(lhs, deltaL);

          llvm::Value *cmpR = builder.CreateFCmpOLT(pRhs, pLhs);
          llvm::Value *deltaR = builder.CreateSelect(cmpR, seed, zero);
          addAdj(rhs, deltaR);
          break;
        }

        case OpCode::MAX: {
          const index_t rhs = tape.operands[--arg_idx];
          const index_t lhs = tape.operands[--arg_idx];
          llvm::Value *pLhs = loadPrimal(lhs);
          llvm::Value *pRhs = loadPrimal(rhs);
          llvm::Value *zero = llvm::ConstantFP::get(scalarTy, 0.0);

          llvm::Value *cmpL = builder.CreateFCmpOGT(pLhs, pRhs);
          llvm::Value *deltaL = builder.CreateSelect(cmpL, seed, zero);
          addAdj(lhs, deltaL);

          llvm::Value *cmpR = builder.CreateFCmpOGT(pRhs, pLhs);
          llvm::Value *deltaR = builder.CreateSelect(cmpR, seed, zero);
          addAdj(rhs, deltaR);
          break;
        }

        case OpCode::POW: {
          const index_t rhs = tape.operands[--arg_idx];
          const index_t lhs = tape.operands[--arg_idx];
          llvm::Value *a = loadPrimal(lhs);
          llvm::Value *b = loadPrimal(rhs);

          llvm::Value *one = llvm::ConstantFP::get(scalarTy, 1.0);
          llvm::Value *b_minus_1 = builder.CreateFSub(b, one);

          llvm::Function *powFn = llvm::Intrinsic::getOrInsertDeclaration(
              &mod, llvm::Intrinsic::pow, {scalarTy});
          llvm::Value *a_pow_bm1 = builder.CreateCall(powFn, {a, b_minus_1});
          llvm::Value *deriv_a = builder.CreateFMul(b, a_pow_bm1);
          addAdj(lhs, builder.CreateFMul(deriv_a, seed));

          llvm::Function *logFn = llvm::Intrinsic::getOrInsertDeclaration(
              &mod, llvm::Intrinsic::log, {scalarTy});
          llvm::Value *log_a = builder.CreateCall(logFn, {a});
          llvm::Value *deriv_b = builder.CreateFMul(loadPrimal(i), log_a);
          addAdj(rhs, builder.CreateFMul(deriv_b, seed));
          break;
        }
      }
    }

    builder.CreateRetVoid();
  }
};

}  // namespace jit
}  // namespace adventure

#endif  // ADVENTURE_ENABLE_LLVM_JIT
