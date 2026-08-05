# ai — build rules
#
# The metallib sync exists because its absence silently disables work you have
# already done. mongoose loads *.metallib from the working directory before
# falling back to inline source, so a stale copy here overrides a freshly built
# kernel in ../mongoose/kernels and the change appears to have no effect.
#
# That is not hypothetical: ai/infer.metallib was found stale (missing an
# attention rewrite) and mlp_train.metallib was missing entirely, which made
# mtl_mlp_gemm_bt return silently — a GEMM producing all zeros, which read as a
# kernel bug rather than a missing file.
#
# Run `make kernels` after any change under ../mongoose/kernels.

MONGOOSE   := ../mongoose
KERNEL_SRC := $(MONGOOSE)/kernels
LIBS       := infer.metallib fused_train.metallib gemm_metal4.metallib \
              mlp_train.metallib sq4_matvec.metallib sq4_gemm_metal4.metallib

.PHONY: all kernels test clean verify-kernels

all: kernels
	go build ./...

# Rebuild mongoose's metallibs from source, then copy them here. Copying without
# rebuilding would just propagate whatever staleness already exists.
kernels:
	@$(MAKE) -C $(MONGOOSE) kernels
	@for lib in $(LIBS); do \
		if [ -f "$(KERNEL_SRC)/$$lib" ]; then \
			cp "$(KERNEL_SRC)/$$lib" .; \
			echo "synced $$lib ($$(md5 -q $$lib))"; \
		else \
			echo "MISSING: $(KERNEL_SRC)/$$lib"; \
		fi; \
	done

# verify-kernels fails when a local metallib differs from mongoose's. A stale
# metallib is a silent correctness bug, so this belongs in CI.
verify-kernels:
	@rc=0; for lib in $(LIBS); do \
		if [ ! -f "$$lib" ]; then echo "MISSING: $$lib"; rc=1; \
		elif [ -f "$(KERNEL_SRC)/$$lib" ] && \
		     [ "$$(md5 -q $$lib)" != "$$(md5 -q $(KERNEL_SRC)/$$lib)" ]; then \
			echo "STALE: $$lib differs from $(KERNEL_SRC)/$$lib"; rc=1; \
		fi; \
	done; \
	if [ $$rc -eq 0 ]; then echo "kernels up to date"; fi; \
	exit $$rc

test: kernels
	go test ./...

clean:
	rm -f $(LIBS)
