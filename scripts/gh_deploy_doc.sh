#!/bin/bash
# NOTE: this script is triggered by github action automatically
# when megred into main

set -euxo pipefail

cd docs
make html
mv _build/html/* .
rm -rf _build/
cd ..

git grep --cached -l '' | xargs rm
git fetch
git checkout -B gh-pages origin/gh-pages
echo "3rdparty/" >> .gitignore

DATE=`date`
git add . && git commit -am "Build at ${DATE}"
git push origin gh-pages
git checkout main && git submodule update
echo "Finish deployment at ${DATE}"

