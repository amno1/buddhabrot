# Makefile for Buddhabrot AVX-512 Generator

SBCL ?= $(firstword $(wildcard ../../sb-simd-512/run-sbcl.sh ../sb-simd-512/run-sbcl.sh $(HOME)/repos/sb-simd-512/run-sbcl.sh) sbcl)
DYNAMIC_SPACE ?= 8000
THREADS ?= 16
SAMPLES ?= 5000

SCRIPT_512 := buddha-512.lisp
FASL_512   := buddha-512.fasl
SCRIPT_DENOISE := denoise.lisp
FASL_DENOISE   := denoise.fasl

OUTPUT          := buddhabrot.exr
OUTPUT_DENOISED := buddhabrot-denoised.exr

.PHONY: all buddha buddha-denoise buddha-512 run compile clean clean-all test test-threads test-registers help

all: buddha
run: buddha

$(FASL_512): $(SCRIPT_512)
	@echo "Compiling $(SCRIPT_512) -> $(FASL_512) with SBCL AVX-512..."
	$(SBCL) --dynamic-space-size $(DYNAMIC_SPACE) --noinform \
		--eval '(progn (compile-file "$<") (sb-ext:exit))'

$(FASL_DENOISE): $(SCRIPT_DENOISE)
	@echo "Compiling $(SCRIPT_DENOISE) -> $(FASL_DENOISE)..."
	$(SBCL) --dynamic-space-size $(DYNAMIC_SPACE) --noinform \
		--eval '(progn (compile-file "$<") (sb-ext:exit))'

compile: $(FASL_512) $(FASL_DENOISE)

buddha: $(FASL_512)
	$(SBCL) --dynamic-space-size $(DYNAMIC_SPACE) --noinform \
		--load $(FASL_512) \
		--eval '(progn (buddha:main $(THREADS) $(SAMPLES) nil "$(OUTPUT)") (sb-ext:exit))'

buddha-denoise: $(FASL_512) $(FASL_DENOISE)
	$(SBCL) --dynamic-space-size $(DYNAMIC_SPACE) --noinform \
		--load $(FASL_DENOISE) \
		--load $(FASL_512) \
		--eval '(progn (buddha:main-denoise $(THREADS) $(SAMPLES) "$(OUTPUT_DENOISED)") (sb-ext:exit))'

buddha-512: buddha

test: compile
	@./tests/run-tests.sh all

test-threads: compile
	@./tests/run-tests.sh threads

test-registers: compile
	@./tests/run-tests.sh registers

clean:
	rm -f $(FASL_512) $(FASL_DENOISE) $(OUTPUT) $(OUTPUT_DENOISED)

clean-all:
	rm -f $(FASL_512) $(FASL_DENOISE) $(OUTPUT) $(OUTPUT_DENOISED) *.ppm *.exr

help:
	@echo "Buddhabrot Makefile"
	@echo "--------------------"
	@echo "Usage:"
	@echo "  make buddha                          Render raw Buddhabrot (5,000 samples/thread)"
	@echo "  make buddha SAMPLES=10000            Render with 10,000 samples/thread (160,000 target orbits)"
	@echo "  make buddha SAMPLES=100000           Render with 100,000 samples/thread (1.6M target orbits)"
	@echo "  make buddha-denoise                  Render Buddhabrot with GPU OIDN denoising in its own thread"
	@echo "  make buddha-denoise SAMPLES=10000    Denoised render with 10,000 samples/thread"
	@echo "  make buddha THREADS=32               Set thread count"
	@echo "  make compile                         Compile both buddha-512.lisp and denoise.lisp"
	@echo "  make test                            Run all benchmarks (threads and registers)"
	@echo "  make test-threads                    Benchmark thread scaling (4 to 20 threads)"
	@echo "  make test-registers                  Benchmark register saturation & ILP (1x, 2x, 3x)"
	@echo "  make clean                           Remove FASLs and output EXRs"
