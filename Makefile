CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Iinclude -Istrategies
TESTFLAGS := $(CXXFLAGS) -Itests

BINDIR := bin
APPS   := run_backtest benchmark validate_fidelity calibrate_hawkes
TESTS  := test_order_book test_matching test_friction test_hawkes

.PHONY: all apps tests check clean

all: apps tests

apps: $(addprefix $(BINDIR)/,$(APPS))

tests: $(addprefix $(BINDIR)/,$(TESTS))

$(BINDIR)/%: apps/%.cpp | $(BINDIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BINDIR)/%: tests/%.cpp | $(BINDIR)
	$(CXX) $(TESTFLAGS) $< -o $@

$(BINDIR):
	mkdir -p $(BINDIR)

check: tests
	@set -e; for t in $(TESTS); do echo "== $$t =="; ./$(BINDIR)/$$t; done

clean:
	rm -rf $(BINDIR) build equity_curve.csv
