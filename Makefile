# Galaxy two-point angular correlation - CUDA (Blackwell) and HIP (RDNA 2).
#
# Single entry point for building, running, benchmarking and checking.
# Everything is overridable:
#
#   make cuda ARCH=sm_89           build for a different NVIDIA target
#   make hip OFFLOAD_ARCH=gfx1100  build for a different AMD target
#   make bench BACKEND=cuda        the published measurement protocol
#   make verify                    reproduce results/omega.out byte for byte
#
# Requires nvcc for the CUDA path and hipcc for the HIP path; each target needs
# only its own compiler. The quality targets need neither - they live in
# Makefile.pre-commit and are delegated to from here, so that the same checks
# run on a machine with no GPU toolchain at all. See CONTRIBUTING.md.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

ROOT_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
SRC_DIR  := $(ROOT_DIR)/src
DATA_DIR := $(ROOT_DIR)/data
BIN_DIR  := $(ROOT_DIR)/bin
RES_DIR  := $(ROOT_DIR)/results
SCRIPTS  := $(ROOT_DIR)/scripts

DATA      := $(DATA_DIR)/data_100k_arcmin.txt
FLAT      := $(DATA_DIR)/flat_100k_arcmin.txt
REFERENCE := $(RES_DIR)/omega.out

BOLD   := \033[1m
GREEN  := \033[32m
YELLOW := \033[33m
CYAN   := \033[36m
RED    := \033[31m
RESET  := \033[0m

PROJECT_NAME := Galaxy two-point angular correlation

# ---- Toolchain -------------------------------------------------------------
NVCC  ?= nvcc
HIPCC ?= hipcc

# NVIDIA. sm_120 is Blackwell, which is what the published numbers were measured on.
ARCH       ?= sm_120
TILE       ?= 512
NVCC_FLAGS ?= -O3 --use_fast_math -arch=$(ARCH) -lineinfo --ptxas-options=-v -DTILE=$(TILE)

# AMD. gfx1030 is the RX 6900 XT. This used to be gated behind a hostname check,
# which meant nobody else could reproduce the AMD numbers.
OFFLOAD_ARCH ?= gfx1030
BLOCK_SIZE_X ?= 32
BLOCK_SIZE_Y ?= 32
HIP_FLAGS    ?= -O3 -ffast-math -munsafe-fp-atomics \
                -DBLOCK_SIZE_X=$(BLOCK_SIZE_X) -DBLOCK_SIZE_Y=$(BLOCK_SIZE_Y) \
                --offload-arch=$(OFFLOAD_ARCH) -mwavefrontsize64

CUDA_BIN := $(BIN_DIR)/galaxy_cuda.out
HIP_BIN  := $(BIN_DIR)/galaxy_hip.out

# The compiler flags are build inputs too, but make only tracks files. These
# stamps put the flags into the dependency graph, so changing ARCH, TILE or
# OFFLOAD_ARCH forces a rebuild instead of silently leaving a binary built for
# the previous target in place. FORCE makes the stamp rule always run; the cmp
# keeps its mtime unchanged when the flags have not actually moved.
CUDA_STAMP := $(BIN_DIR)/.cuda-flags
HIP_STAMP  := $(BIN_DIR)/.hip-flags

# Which backend the generic run/bench/verify targets use. Defaults to whichever
# compiler is installed, preferring CUDA.
BACKEND ?= $(shell command -v $(NVCC) >/dev/null 2>&1 && echo cuda || echo hip)
BACKEND_BIN = $(if $(filter cuda,$(BACKEND)),$(CUDA_BIN),$(HIP_BIN))

# Number of timed runs. The published protocol uses 9: run 1 is cold and is
# reported separately, the median is taken over runs 2-9.
RUNS       ?= 9
QUICK_RUNS ?= 5

IMAGE    := galaxy-correlation:dev
PC_IMAGE := galaxy-correlation-pre-commit:local

.PHONY: FORCE all cuda hip run run-cuda run-hip check-env bench verify spill-check \
        format format-check lint tidy data-integrity \
        pre-commit pre-commit-build install-hooks docker docker-shell \
        clean info help
.DEFAULT_GOAL := help

##@ Help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\n$(BOLD)$(PROJECT_NAME)$(RESET)\n"} \
	      /^[a-zA-Z_.-]+:.*?##/ { printf "  $(CYAN)%-18s$(RESET) %s\n", $$1, $$2 } \
	      /^##@/ { printf "\n$(BOLD)%s$(RESET)\n", substr($$0, 5) }' $(MAKEFILE_LIST)
	@printf "\n$(BOLD)Current settings$(RESET)\n"
	@printf "  Default backend    $(BACKEND)   (override with BACKEND=cuda|hip)\n"
	@printf "  NVIDIA target      ARCH=$(ARCH)  TILE=$(TILE)\n"
	@printf "  AMD target         OFFLOAD_ARCH=$(OFFLOAD_ARCH)  BLOCK=$(BLOCK_SIZE_X)x$(BLOCK_SIZE_Y)\n"
	@printf "  Benchmark runs     RUNS=$(RUNS)\n\n"

##@ Build
all: ## Build every backend whose compiler is available
	@built=0; \
	if command -v $(NVCC) >/dev/null 2>&1; then $(MAKE) --no-print-directory cuda; built=1; fi; \
	if command -v $(HIPCC) >/dev/null 2>&1; then $(MAKE) --no-print-directory hip; built=1; fi; \
	if [ "$$built" -eq 0 ]; then \
		printf "$(RED)error: neither $(NVCC) nor $(HIPCC) found in PATH$(RESET)\n" >&2; exit 1; \
	fi

cuda: $(CUDA_BIN) ## Build the CUDA binary (NVIDIA)

$(CUDA_BIN): $(SRC_DIR)/galaxy_cuda.cu $(CUDA_STAMP) | $(BIN_DIR)
	@command -v $(NVCC) >/dev/null 2>&1 || { printf "$(RED)error: $(NVCC) not found in PATH$(RESET)\n" >&2; exit 1; }
	$(NVCC) $(NVCC_FLAGS) $< -o $@
	@printf "$(GREEN)built $@ for $(ARCH), TILE=$(TILE)$(RESET)\n"

hip: $(HIP_BIN) ## Build the HIP binary (AMD)

$(HIP_BIN): $(SRC_DIR)/galaxy_hip.cpp $(HIP_STAMP) | $(BIN_DIR)
	@command -v $(HIPCC) >/dev/null 2>&1 || { printf "$(RED)error: $(HIPCC) not found in PATH$(RESET)\n" >&2; exit 1; }
	$(HIPCC) $(HIP_FLAGS) $< -o $@
	@printf "$(GREEN)built $@ for $(OFFLOAD_ARCH)$(RESET)\n"

$(BIN_DIR) $(RES_DIR):
	@mkdir -p $@

$(CUDA_STAMP): FORCE | $(BIN_DIR)
	@echo '$(NVCC_FLAGS)' | cmp -s - $@ || echo '$(NVCC_FLAGS)' > $@

$(HIP_STAMP): FORCE | $(BIN_DIR)
	@echo '$(HIP_FLAGS)' | cmp -s - $@ || echo '$(HIP_FLAGS)' > $@

##@ Run
run: $(BACKEND) ## Build and run the default backend a few times
	@$(SCRIPTS)/run-series.sh $(BACKEND_BIN) $(QUICK_RUNS) $(RES_DIR)/omega.out $(DATA) $(FLAT)

run-cuda: cuda ## Build and run the CUDA binary
	@$(SCRIPTS)/run-series.sh $(CUDA_BIN) $(QUICK_RUNS) $(RES_DIR)/omega.out $(DATA) $(FLAT)

run-hip: hip ## Build and run the HIP binary
	@$(SCRIPTS)/run-series.sh $(HIP_BIN) $(QUICK_RUNS) $(RES_DIR)/omega.out $(DATA) $(FLAT)

##@ Benchmarking
check-env: ## Warn about anything that would skew a benchmark
	@$(SCRIPTS)/check-env.sh

bench: $(BACKEND) check-env ## The published measurement protocol
	@printf "$(BOLD)==> Benchmarking $(BACKEND) over $(RUNS) runs.$(RESET)\n"
	@printf "    Run 1 is cold and is reported separately; the figure to quote\n"
	@printf "    is the median of runs 2-$(RUNS).\n\n"
	@$(SCRIPTS)/run-series.sh $(BACKEND_BIN) $(RUNS) $(RES_DIR)/omega.bench.out $(DATA) $(FLAT)

# Turns "34 registers, zero spilled" from a claim into an assertion - and one
# that holds on a machine with no GPU, since ptxas runs at compile time.
spill-check: | $(BIN_DIR) ## Assert the kernel still compiles with zero register spills
	@command -v $(NVCC) >/dev/null 2>&1 || { printf "$(RED)error: $(NVCC) not found in PATH$(RESET)\n" >&2; exit 1; }
	@out=$$($(NVCC) $(NVCC_FLAGS) $(SRC_DIR)/galaxy_cuda.cu -o $(CUDA_BIN) 2>&1); \
	echo "$$out" | grep -E 'Used [0-9]+ registers|spill' || true; \
	if ! echo "$$out" | grep -q '0 bytes spill stores, 0 bytes spill loads'; then \
		printf "$(RED)FAIL: ptxas reports register spills for $(ARCH) TILE=$(TILE)$(RESET)\n" >&2; exit 1; \
	fi; \
	regs=$$(echo "$$out" | grep -oE 'Used [0-9]+ registers' | grep -oE '[0-9]+' | head -1); \
	printf "$(GREEN)OK: $(ARCH) TILE=$(TILE) - $$regs registers, zero spills$(RESET)\n"

##@ Correctness
verify: $(BACKEND) | $(RES_DIR) ## Run once and check the output matches the committed reference
	@test -f $(REFERENCE) || { printf "$(RED)error: missing reference $(REFERENCE)$(RESET)\n" >&2; exit 1; }
	@printf "$(BOLD)==> Running $(BACKEND) and comparing against the committed reference.$(RESET)\n"
	@$(BACKEND_BIN) $(DATA) $(FLAT) $(RES_DIR)/omega.verify.out >/dev/null
	@if diff -q $(REFERENCE) $(RES_DIR)/omega.verify.out >/dev/null; then \
		printf "$(GREEN)OK: output matches $(REFERENCE) byte for byte.$(RESET)\n"; \
	else \
		printf "$(RED)FAIL: output differs from $(REFERENCE).$(RESET)\n" >&2; \
		diff -u $(REFERENCE) $(RES_DIR)/omega.verify.out | head -20 >&2; \
		exit 1; \
	fi

data-integrity: ## Check the catalogs and reference output are byte-unchanged
	@$(MAKE) --no-print-directory -f Makefile.pre-commit data-integrity

##@ Code Quality
# Implemented in Makefile.pre-commit, which deliberately needs no GPU toolchain
# so the same checks run in CI and in the pre-commit container. See that file.
format: ## Apply clang-format in place
	@$(MAKE) --no-print-directory -f Makefile.pre-commit format

format-check: ## Fail if any source is not formatted
	@$(MAKE) --no-print-directory -f Makefile.pre-commit format-check

lint: ## Full no-GPU gate (format, shell, docker, secrets, data, links)
	@$(MAKE) --no-print-directory -f Makefile.pre-commit lint

tidy: ## clang-tidy (slow; needs CUDA/ROCm headers)
	@$(MAKE) --no-print-directory -f Makefile.pre-commit tidy

install-hooks: ## Install the pre-commit hooks into .git/hooks
	@command -v pre-commit >/dev/null 2>&1 || { printf "$(RED)error: pre-commit not installed (pip install pre-commit)$(RESET)\n" >&2; exit 1; }
	@pre-commit install --install-hooks
	@printf "$(GREEN)hooks installed$(RESET)\n"

##@ Containers
pre-commit-build: ## Build the CUDA-free image the hooks run in
	docker build -f $(ROOT_DIR)/Dockerfile.pre-commit -t $(PC_IMAGE) $(ROOT_DIR)

pre-commit: pre-commit-build ## Run every hook inside that image
	docker run --rm \
	  -v "$(ROOT_DIR)":/src \
	  -v pre-commit-cache:/root/.cache/pre-commit \
	  $(PC_IMAGE) run --all-files

docker: ## Build the CUDA development image
	docker build -t $(IMAGE) $(ROOT_DIR)

docker-shell: docker ## Open a shell in the development image
	docker run --rm -it --gpus all -v "$(ROOT_DIR)":/work -w /work $(IMAGE) bash

##@ Housekeeping
clean: ## Remove binaries and run artifacts
	@rm -rf $(BIN_DIR)
	@rm -f $(RES_DIR)/omega.bench.out $(RES_DIR)/omega.verify.out $(RES_DIR)/*.previous
	@printf "$(GREEN)clean$(RESET)\n"

info: ## Print the resolved build configuration
	@printf "$(BOLD)$(PROJECT_NAME)$(RESET)\n\n"
	@printf "  Backend        : $(BACKEND)\n"
	@printf "  nvcc           : $$(command -v $(NVCC) 2>/dev/null || echo '(not found)')\n"
	@printf "  hipcc          : $$(command -v $(HIPCC) 2>/dev/null || echo '(not found)')\n"
	@printf "  NVCC_FLAGS     : $(NVCC_FLAGS)\n"
	@printf "  HIP_FLAGS      : $(HIP_FLAGS)\n"
	@printf "  Data           : $(DATA)\n"
	@printf "                   $(FLAT)\n"
	@printf "  Reference      : $(REFERENCE)\n"
	@printf "  Benchmark runs : $(RUNS) (median over runs 2-$(RUNS))\n"

.SILENT: info help

FORCE:
