#!/bin/bash
# NOTE: this script is triggered by github action automatically
# when megred into main

set -euxo pipefail

cd docs
make html
mv _build/html/* .
rm -rf _build/
cd ..

git ls-file | xargs rm
git fetch
git checkout -B gh-pages origin/gh-pages

DATE=`date`
git add html/ && git commit -am "Build at ${DATE}"
git push origin gh-pages
git checkout main && git submodule update
echo "Finish deployment at ${DATE}"
