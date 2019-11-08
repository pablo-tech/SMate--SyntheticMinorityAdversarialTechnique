#!/usr/bin/env bash
# $1 [branchname]

git branch $1 || exit 1
git checkout $1 || exit 1
git push --set-upstream origin $1 || exit 1
