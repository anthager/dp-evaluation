#!/usr/bin/env bash
set -e

pg_ctl -D $PGDATA -w start

psql <<-'EOSQL'
    create extension anon_func;
EOSQL

pg_ctl -w stop

# starts in the forground
postgres -D $PGDATA 2>&1
