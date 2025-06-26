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

def cancel_previous_build() {
  // cancel previous build if it is not on main.
  if (env.BRANCH_NAME != 'main') {
    def buildNumber = env.BUILD_NUMBER as int
    // Milestone API allows us to cancel previous build
    // with the same milestone number
    if (buildNumber > 1) milestone(buildNumber - 1)
    milestone(buildNumber)
  }
}

def is_last_build() {
  // whether it is last build
  def job = Jenkins.instance.getItem(env.JOB_NAME)
  def lastBuild = job.getLastBuild()
  return lastBuild.getNumber() == env.BUILD_NUMBER
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

def run_unittest_CPU_AOT_COMPILE(node_type) {
  echo "Running CPU AOT Compile Unittest"

  if (node_type.contains('SPOT')) {
    // For SPOT: timeout only node allocation to prevent hanging
    timeout(time: 10, unit: 'MINUTES') {
      node(node_type) {
        // Node successfully allocated, now run tests without time limit
        ws(per_exec_ws('flashinfer-aot')) {
          init_git(true)
          sh(script: "ls -alh", label: 'Show work directory')
          sh(script: "./scripts/task_show_node_info.sh", label: 'Show node info')
          sh(script: "${docker_run} --no-gpu ./scripts/task_test_aot_build_import.sh", label: 'Test AOT Build and Import')
        }
      }
    }
  } else {
    node(node_type) {
      ws(per_exec_ws('flashinfer-aot')) {
        init_git(true)
        sh(script: "ls -alh", label: 'Show work directory')
        sh(script: "./scripts/task_show_node_info.sh", label: 'Show node info')
        sh(script: "${docker_run} --no-gpu ./scripts/task_test_aot_build_import.sh", label: 'Test AOT Build and Import')
      }
    }
  }
}

def shard_run_unittest_GPU(node_type, shard_id) {
  echo "Running unittest on ${node_type}, shard ${shard_id}"
  node(node_type) {
    ws(per_exec_ws('flashinfer-unittest')) {
      init_git(true) // we need cutlass submodule
      sh(script: "ls -alh", label: 'Show work directory')
      sh(script: "./scripts/task_show_node_info.sh", label: 'Show node info')
      sh(script: "${docker_run} ./scripts/task_jit_run_tests_part${shard_id}.sh", label: 'JIT Unittest Part ${shard_id}')
    }
  }
}

stage('Unittest') {
  cancel_previous_build()
  parallel(
    failFast: true,
    'AOT-Build-Import': {
      try {
        run_unittest_CPU_AOT_COMPILE('CPU-LARGE-SPOT')
      } catch (Throwable ex) {
          def isLast = is_last_build()
          echo "Exception during SPOT run: ${ex.toString()}"
          echo "Current build: ${env.BUILD_NUMBER}, is_last_build(): ${isLast}"

          if (isLast) {
            // retry if we are currently at last build
            // mark the current stage as success
            // and try again via on demand node
            echo 'Retrying with CPU-LARGE because this is the last build'
            currentBuild.result = 'SUCCESS'
            run_unittest_CPU_AOT_COMPILE('CPU-LARGE')
          } else {
            echo 'Not retrying because this is not the last build'
            throw ex
          }
      }
    },
    'JIT-Unittest-1': {
      try {
        shard_run_unittest_GPU('GPU-G5-SPOT', 1)
      } catch (Throwable ex) {
        if (is_last_build()) {
          // retry if we are currently at last build
          // mark the current stage as success
          // and try again via on demand node
          echo 'Exception during SPOT run ' + ex.toString() + ' retry on-demand'
          currentBuild.result = 'SUCCESS'
          shard_run_unittest_GPU('GPU-G5', 1)
        } else {
          echo 'Exception during SPOT run ' + ex.toString() + ' exit since it is not last build'
          throw ex
        }
      }
    },
    'JIT-Unittest-2': {
      try {
        shard_run_unittest_GPU('GPU-G5-SPOT', 2)
      } catch (Throwable ex) {
        if (is_last_build()) {
          // retry if we are currently at last build
          // mark the current stage as success
          // and try again via on demand node
          echo 'Exception during SPOT run ' + ex.toString() + ' retry on-demand'
          currentBuild.result = 'SUCCESS'
          shard_run_unittest_GPU('GPU-G5', 2)
        } else {
          echo 'Exception during SPOT run ' + ex.toString() + ' exit since it is not last build'
          throw ex
        }
      }
    },
    'JIT-Unittest-3': {
      try {
        shard_run_unittest_GPU('GPU-G5-SPOT', 3)
      } catch (Throwable ex) {
        if (is_last_build()) {
          // retry if we are currently at last build
          // mark the current stage as success
          // and try again via on demand node
          echo 'Exception during SPOT run ' + ex.toString() + ' retry on-demand'
          currentBuild.result = 'SUCCESS'
          shard_run_unittest_GPU('GPU-G5', 3)
        } else {
          echo 'Exception during SPOT run ' + ex.toString() + ' exit since it is not last build'
          throw ex
        }
      }
    },
    'JIT-Unittest-4': {
      try {
        shard_run_unittest_GPU('GPU-G5-SPOT', 4)
      } catch (Throwable ex) {
        if (is_last_build()) {
          // retry if we are currently at last build
          // mark the current stage as success
          // and try again via on demand node
          echo 'Exception during SPOT run ' + ex.toString() + ' retry on-demand'
          currentBuild.result = 'SUCCESS'
          shard_run_unittest_GPU('GPU-G5', 4)
        } else {
          echo 'Exception during SPOT run ' + ex.toString() + ' exit since it is not last build'
          throw ex
        }
      }
    }
  )
}
