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
// Now supports multiple CUDA versions
docker_run_cu126 = "bash ci/bash.sh flashinfer/flashinfer-ci-cu126:latest"
docker_run_cu128 = "bash ci/bash.sh flashinfer/flashinfer-ci-cu128:latest"
docker_run_cu129 = "bash ci/bash.sh flashinfer/flashinfer-ci-cu129:latest"
docker_run_cu130 = "bash ci/bash.sh flashinfer/flashinfer-ci-cu130:latest"

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
  // check whether it is last build
  try {
    return currentBuild.number == currentBuild.rawBuild.project.getLastBuild().number
  } catch (Throwable ex) {
    echo 'Error during check is_last_build ' + ex.toString()
    return false
  }
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

def run_with_spot_retry(spot_node_type, on_demand_node_type, test_name, test_closure) {
  try {
    test_closure(spot_node_type)
  } catch (hudson.AbortException abortEx) {
    echo "Received normal AbortException, exit now: " + abortEx.toString()
    throw abortEx
  } catch (Throwable ex) {
    echo "Exception during SPOT run for ${test_name}: " + ex.toString()
    if (is_last_build()) {
      echo "Exception during SPOT run for ${test_name}: " + ex.toString() + " retry on-demand"
      currentBuild.result = 'SUCCESS'
      test_closure(on_demand_node_type)
    } else {
      echo 'Exit since it is not last build'
      throw ex
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

def run_unittest_CPU_AOT_COMPILE(node_type, cuda_version) {
  echo "Running CPU AOT Compile Unittest with CUDA ${cuda_version}"

  def docker_run = ""
  if (cuda_version == "cu126") {
    docker_run = docker_run_cu126
  } else if (cuda_version == "cu128") {
    docker_run = docker_run_cu128
  } else if (cuda_version == "cu129") {
    docker_run = docker_run_cu129
  } else if (cuda_version == "cu130") {
    docker_run = docker_run_cu130
  } else {
    error("Unknown CUDA version: ${cuda_version}")
  }

  if (node_type.contains('SPOT')) {
    // Add timeout only for spot instances - node allocation only
    def node_allocated = false

    try {
      timeout(time: 15, unit: 'MINUTES') {
        // Only timeout the node allocation, not the test execution
        node(node_type) {
          node_allocated = true
          // Just mark that we got the node, don't run tests here
        }
      }

      // If we reach here, node allocation was successful
      // Now run the tests without any timeout
      node(node_type) {
        ws(per_exec_ws('flashinfer-aot')) {
          init_git(true)
          sh(script: "ls -alh", label: 'Show work directory')
          sh(script: "./scripts/task_show_node_info.sh", label: 'Show node info')
          sh(script: "${docker_run} --no-gpu ./scripts/task_test_aot_build_import.sh", label: 'Test AOT Build and Import')
        }
      }
    } catch (Exception e) {
      if (!node_allocated) {
        echo "Node allocation timeout or failure after 15 minutes for ${node_type}: ${e.toString()}"
      }
      throw e
    }
  } else {
    // No timeout for non-spot instances
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

def shard_run_unittest_GPU(node_type, shard_id, cuda_version) {
  echo "Running unittest on ${node_type}, shard ${shard_id}, CUDA ${cuda_version}"

  def docker_run = ""
  if (cuda_version == "cu126") {
    docker_run = docker_run_cu126
  } else if (cuda_version == "cu128") {
    docker_run = docker_run_cu128
  } else if (cuda_version == "cu129") {
    docker_run = docker_run_cu129
  } else {
    error("Unknown CUDA version: ${cuda_version}")
  }

  if (node_type.contains('SPOT')) {
    // Add timeout only for spot instances - node allocation only
    def node_allocated = false

    try {
      timeout(time: 15, unit: 'MINUTES') {
        // Only timeout the node allocation, not the test execution
        node(node_type) {
          node_allocated = true
          // Just mark that we got the node, don't run tests here
        }
      }

      // If we reach here, node allocation was successful
      // Now run the tests without any timeout
      node(node_type) {
        ws(per_exec_ws('flashinfer-unittest')) {
          init_git(true) // we need cutlass submodule
          sh(script: "ls -alh", label: 'Show work directory')
          sh(script: "./scripts/task_show_node_info.sh", label: 'Show node info')
          sh(script: "${docker_run} ./scripts/task_jit_run_tests_part${shard_id}.sh", label: 'JIT Unittest Part ${shard_id}')
        }
      }
    } catch (Exception e) {
      if (!node_allocated) {
        echo "Node allocation timeout or failure after 15 minutes for ${node_type}: ${e.toString()}"
      }
      throw e
    }
  } else {
    // No timeout for non-spot instances
    node(node_type) {
      ws(per_exec_ws('flashinfer-unittest')) {
        init_git(true) // we need cutlass submodule
        sh(script: "ls -alh", label: 'Show work directory')
        sh(script: "./scripts/task_show_node_info.sh", label: 'Show node info')
        sh(script: "${docker_run} ./scripts/task_jit_run_tests_part${shard_id}.sh", label: 'JIT Unittest Part ${shard_id}')
      }
    }
  }
}

stage('Unittest') {
  cancel_previous_build()
  parallel(
    failFast: true,
    // CUDA 12.6 AOT Tests
    'AOT-Build-Import-x86-64-cu126': {
      run_with_spot_retry('CPU-LARGE-SPOT', 'CPU-LARGE', 'AOT-Build-Import-x86-64-cu126',
        { node_type -> run_unittest_CPU_AOT_COMPILE(node_type, 'cu126') })
    },
    'AOT-Build-Import-aarch64-cu126': {
      run_with_spot_retry('ARM-LARGE-SPOT', 'ARM-LARGE', 'AOT-Build-Import-aarch64-cu126',
        { node_type -> run_unittest_CPU_AOT_COMPILE(node_type, 'cu126') })
    },
    // CUDA 12.8 AOT Tests
    'AOT-Build-Import-x86-64-cu128': {
      run_with_spot_retry('CPU-LARGE-SPOT', 'CPU-LARGE', 'AOT-Build-Import-x86-64-cu128',
        { node_type -> run_unittest_CPU_AOT_COMPILE(node_type, 'cu128') })
    },
    'AOT-Build-Import-aarch64-cu128': {
      run_with_spot_retry('ARM-LARGE-SPOT', 'ARM-LARGE', 'AOT-Build-Import-aarch64-cu128',
        { node_type -> run_unittest_CPU_AOT_COMPILE(node_type, 'cu128') })
    },
    // CUDA 12.9 AOT Tests
    'AOT-Build-Import-x86-64-cu129': {
      run_with_spot_retry('CPU-LARGE-SPOT', 'CPU-LARGE', 'AOT-Build-Import-x86-64-cu129',
        { node_type -> run_unittest_CPU_AOT_COMPILE(node_type, 'cu129') })
    },
    'AOT-Build-Import-aarch64-cu129': {
      run_with_spot_retry('ARM-LARGE-SPOT', 'ARM-LARGE', 'AOT-Build-Import-aarch64-cu129',
        { node_type -> run_unittest_CPU_AOT_COMPILE(node_type, 'cu129') })
    },
    // CUDA 13.0 AOT Tests
    'AOT-Build-Import-x86-64-cu130': {
      run_with_spot_retry('CPU-LARGE-SPOT', 'CPU-LARGE', 'AOT-Build-Import-x86-64-cu130',
        { node_type -> run_unittest_CPU_AOT_COMPILE(node_type, 'cu130') })
    },
    'AOT-Build-Import-aarch64-cu130': {
      run_with_spot_retry('ARM-LARGE-SPOT', 'ARM-LARGE', 'AOT-Build-Import-aarch64-cu130',
        { node_type -> run_unittest_CPU_AOT_COMPILE(node_type, 'cu130') })
    },
    // JIT unittest only for cu129
    'JIT-Unittest-1-cu129': {
      run_with_spot_retry('GPU-G5-SPOT', 'GPU-G5', 'JIT-Unittest-1-cu129',
        { node_type -> shard_run_unittest_GPU(node_type, 1, 'cu129') })
    },
    'JIT-Unittest-2-cu129': {
      run_with_spot_retry('GPU-G5-SPOT', 'GPU-G5', 'JIT-Unittest-2-cu129',
        { node_type -> shard_run_unittest_GPU(node_type, 2, 'cu129') })
    },
    'JIT-Unittest-3-cu129': {
      run_with_spot_retry('GPU-G5-SPOT', 'GPU-G5', 'JIT-Unittest-3-cu129',
        { node_type -> shard_run_unittest_GPU(node_type, 3, 'cu129') })
    },
    'JIT-Unittest-4-cu129': {
      run_with_spot_retry('GPU-G5-SPOT', 'GPU-G5', 'JIT-Unittest-4-cu129',
        { node_type -> shard_run_unittest_GPU(node_type, 4, 'cu129') })
    },
    'JIT-Unittest-5-cu129': {
      run_with_spot_retry('GPU-G5-SPOT', 'GPU-G5', 'JIT-Unittest-5-cu129',
        { node_type -> shard_run_unittest_GPU(node_type, 5, 'cu129') })
    },
  )
}
