#!/usr/bin/bash

function table_create() {
    local TABLE_NAME
    local COLUMNS
    TABLE_NAME=$1
    COLUMNS=$2
    echo "$COLUMNS" >"$TABLE_NAME"
}

function table_add_row() {
    local TABLE_NAME
    local ROW
    TABLE_NAME=$1
    ROW=$2
    echo "$ROW" >>"$TABLE_NAME"
}

function where_is_script() {
    local script=$1
    cd "$(dirname "$script")" && pwd
}