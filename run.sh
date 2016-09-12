#!/bin/bash

if [[ x$RUNGDB == x ]]; then {
    python test_cltorch.py
} else {
    rungdb.sh python test_cltorch.py
} fi
