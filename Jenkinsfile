// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// Jenkins pipeline
// See documents at https://jenkins.io/doc/book/pipeline/jenkinsfile/

// Docker env used for testing
// Different image may have different version tag
// because some of them are more stable than anoter.
//
// Docker images are maintained by PMC, cached in dockerhub
// and remains relatively stable over the time.
// Flow for upgrading docker env(need commiter)
//
// - Send PR to upgrade build script in the repo
// - Build the new docker image
// - Tag the docker image with a new version and push to a binary cache.
// - Update the version in the Jenkinsfile, send a PR
// - Fix any issues wrt to the new image version in the PR
// - Merge the PR and now we are in new version
// - Tag the new version as the lates
// - Periodically cleanup the old versions on local workers
//

import org.jenkinsci.plugins.pipeline.modeldefinition.Utils
// These are set at runtime from data in ci/jenkins/docker-images.yml, update
// image tags in that file
docker_run = "bash ci/bash.sh flashinfer/flashinfer-ci:latest"

def per_exec_ws(folder) {
  return "workspace/exec_${env.EXECUTOR_NUMBER}/" + folder
}

def pack_lib(name, libs) {
  sh """
     echo "Packing ${libs} into ${name}"
     echo ${libs} | sed -e 's/,/ /g' | xargs md5sum
     """
  stash includes: libs, name: name
}

def unpack_lib(name, libs) {
  unstash name
  sh """
     echo "Unpacked ${libs} from ${name}"
     echo ${libs} | sed -e 's/,/ /g' | xargs md5sum
     """
}

def init_git(submodule = false) {
  cleanWs()
  // add retry in case checkout timeouts
  retry(5) {
    checkout scm
  }
  if (submodule) {
    retry(5) {
      timeout(time: 10, unit: 'MINUTES') {
        sh(script: 'git submodule update --init --recursive -f', label: 'Update git submodules')
      }
    }
  }
}

// stage('Lint') {
//   node('CPU-SPOT') {
//     ws(per_exec_ws('flashinfer-lint')) {
//       init_git(false)
//     }
//   }
// }

stage('JIT Unittest') {
  parallel(
    failFast: true,
    'GPU-G6-Test-1': {
      node('GPU-G6-SPOT') {
        ws(per_exec_ws('flashinfer-unittest')) {
          init_git(true) // we need cutlass submodule
          sh(script: "ls -alh", label: 'Show work directory')
          sh(script: "./scripts/task_show_node_info.sh", label: 'Show node info')
          sh(script: "${docker_run} ./scripts/task_jit_run_tests_part1.sh", label: 'JIT Unittest Part 1')
        }
      }
    },
    'GPU-G6-Test-2': {
      node('GPU-G6-SPOT') {
        ws(per_exec_ws('flashinfer-unittest')) {
          init_git(true) // we need cutlass submodule
          sh(script: "ls -alh", label: 'Show work directory')
          sh(script: "./scripts/task_show_node_info.sh", label: 'Show node info')
          sh(script: "${docker_run} ./scripts/task_jit_run_tests_part2.sh", label: 'JIT Unittest Part 2')
        }
      }
    },
    'GPU-G6-Test-4': {
      node('GPU-G6-SPOT') {
        ws(per_exec_ws('flashinfer-unittest')) {
          init_git(true) // we need cutlass submodule
          sh(script: "ls -alh", label: 'Show work directory')
          sh(script: "./scripts/task_show_node_info.sh", label: 'Show node info')
          sh(script: "${docker_run} ./scripts/task_jit_run_tests_part4.sh", label: 'JIT Unittest Part 4')
        }
      }
    }
  )
}
