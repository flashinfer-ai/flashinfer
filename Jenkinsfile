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
ci_image = 'flashinfer/flashinfer-ci:latest'

// Parameters to allow overriding (in Jenkins UI), the image
// to be used by a given build. When provided, they take precedence
// over default values above.
properties([
  parameters([
    string(name: 'ci_image_param', defaultValue: ''),
  ])
])

// Global variable assigned during Sanity Check that holds the sha1 which should be
// merged into the PR in all branches.
upstream_revision = null

// command to start a docker container
docker_run = 'docker/bash.sh'
// timeout in minutes
max_time = 180
// Jenkins script root directory
jenkins_scripts_root = "ci/scripts/jenkins"

// General note: Jenkins has limits on the size of a method (or top level code)
// that are pretty strict, so most usage of groovy methods in these templates
// are purely to satisfy the JVM
def per_exec_ws(folder) {
  return "workspace/exec_${env.EXECUTOR_NUMBER}/" + folder
}

// initialize source codes
def init_git() {
  retry(5) {
    checkout scm
  }

  // Determine merge commit to use for all stages
  if (env.BRANCH_NAME == 'main') {
    // Only set upstream_revision to HEAD and skip merging to avoid a race with another commit merged to main.
    update_upstream_revision("HEAD")
  } else {
    // This is PR branch so merge with latest main.
    merge_with_main()
  }

  sh(
    script: """
      set -eux
      . ${jenkins_scripts_root}/retry.sh
      retry 3 timeout 5m git submodule update --init --recursive -f --jobs 0
    """,
    label: 'Update git submodules',
  )
}

def update_upstream_revision(git_ref) {
  if (upstream_revision == null) {
    upstream_revision = sh(
      script: "git log -1 ${git_ref} --format=\'%H\'",
      label: 'Determine upstream revision',
      returnStdout: true,
    ).trim()
  }
}

def merge_with_main() {
  sh (
    script: 'git fetch origin main',
    label: 'Fetch upstream',
  )
  update_upstream_revision("FETCH_HEAD")
  sh (
    script: "git -c user.name=FlashInfer-Jenkins -c user.email=41898282+github-actions[bot]@users.noreply.github.com merge ${upstream_revision}",
    label: 'Merge to origin/main'
  )
}

def docker_init(image) {
  // Clear out all Docker images that aren't going to be used
  sh(
    script: """
    set -eux
    docker image ls --all
    IMAGES=\$(docker image ls --all --format '{{.Repository}}:{{.Tag}}  {{.ID}}')

    echo -e "Found images:\\n\$IMAGES"
    echo "\$IMAGES" | { grep -vE '${image}' || test \$? = 1; } | { xargs docker rmi || test \$? = 123; }

    docker image ls --all
    """,
    label: 'Clean old Docker images',
  )

  sh(
    script: """
    set -eux
    . ${jenkins_scripts_root}/retry.sh
    retry 5 docker pull ${image}
    """,
    label: 'Pull docker image',
  )
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

def should_skip_ci(pr_number) {
  if (env.BRANCH_NAME == null || !env.BRANCH_NAME.startsWith('PR-')) {
    // never skip CI on build sourced from a branch
    return false
  }
  glob_skip_ci_code = sh (
    returnStatus: true,
    script: "./${jenkins_scripts_root}/git_skip_ci_globs.py",
    label: 'Check if CI should be skipped due to changed files',
  )
  if (glob_skip_ci_code == 0) {
    return true
  }
  withCredentials([string(
    credentialsId: 'tvm-bot-jenkins-reader',
    variable: 'GITHUB_TOKEN',
    )]) {
    // Exit code of 1 means run full CI (or the script had an error, so run
    // full CI just in case). Exit code of 0 means skip CI.
    git_skip_ci_code = sh (
      returnStatus: true,
      script: "./${jenkins_scripts_root}/git_skip_ci.py --pr '${pr_number}'",
      label: 'Check if CI should be skipped',
    )
  }
  return git_skip_ci_code == 0
}

def check_pr(pr_number) {
  if (env.BRANCH_NAME == null || !env.BRANCH_NAME.startsWith('PR-')) {
    // never skip CI on build sourced from a branch
    return false
  }
  withCredentials([string(
    credentialsId: 'tvm-bot-jenkins-reader',
    variable: 'GITHUB_TOKEN',
    )]) {
    sh (
      script: "python3 ${jenkins_scripts_root}/check_pr.py --pr ${pr_number}",
      label: 'Check PR title and body',
    )
  }

}

def prepare(node_type) {
  stage('Prepare') {
    node(node_type) {
      ws("workspace/exec_${env.EXECUTOR_NUMBER}/flashinfer/prepare") {
        init_git()

        // Get image parameter if specified
        ci_image = params.ci_image_param ?: ci_image

        sh (script: """
          echo "Docker image being used in this build:"
          echo " ci_image = ${ci_image}"
        """, label: 'Docker image name')

        is_docs_only_build = sh (
          returnStatus: true,
          script: "./${jenkins_scripts_root}/git_change_docs.sh",
          label: 'Check for docs only changes',
        )
        skip_ci = should_skip_ci(env.CHANGE_ID)
      }
    }
  }
}

def ci_setup(image) {
  sh (
    script: "${docker_run} ${image} ./tests/scripts/task_clear_pytest.sh",
    label: 'Clean up old workspace',
  )
}

def run_build(node_type) {
  if (!skip_ci) {
    echo 'Begin running node_type ' + node_type
    node(node_type) {
      ws("workspace/exec_${env.EXECUTOR_NUMBER}/flashinfer/build") {
        init_git()
        docker_init(ci_image)
        timeout(time: max_time, unit: 'MINUTES') {
          withEnv([
            'PLATFORM=gpu',
          ], {
            sh "${docker_run} --no-gpu ${ci_image} ./scripts/build.sh"
          })
        }
      }
    }
    echo 'End running node_type ' + node_type
  } else {
    Utils.markStageSkippedForConditional('BUILD')
  }
}

def build() {
  stage('Build') {
    try {
      run_build('CPU-SPOT')
    } catch (Throwable ex) {
      if (is_last_build()) {
        // retry if we are currently at last build
        // mark the current stage as success
        // and try again via on demand node
        echo 'Exception during SPOT run ' + ex.toString() + ' retry on-demand'
        currentBuild.result = 'SUCCESS'
        run_build('CPU')
      } else {
        echo 'Exception during SPOT run ' + ex.toString() + ' exit since it is not last build'
        throw ex
      }
    }
  }
}

def run_tests(node_type) {
  echo 'Begin running tests on node_type ' + node_type
  if (!skip_ci && is_docs_only_build != 1) {
    node(node_type) {
      ws("workspace/exec_${env.EXECUTOR_NUMBER}/flashinfer/tests") {
        init_git()
        docker_init(ci_image)
        timeout(time: max_time, unit: 'MINUTES') {
          withEnv([
            'PLATFORM=gpu',
            'TEST_STEP_NAME=unittest',
        ], {
            ci_setup(ci_image)
            sh (
              script: "${docker_run} ${ci_image} ./tests/run_tests.sh",
              label: 'Run tests',
            )
          })
        }
        // only run upload if things are successful
        try {
          junit 'build/pytest-results/*.xml'
        } catch (Exception e) {
          echo 'Exception during JUnit upload: ' + e.toString()
        }
      }
    }
    echo 'End running tests on node_type ' + node_type
  } else {
    Utils.markStageSkippedForConditional('TESTS')
  }
}

def test() {
  stage('Test') {
    try {
      run_tests('GPU-SPOT')
    } catch (Throwable ex) {
      if (is_last_build()) {
        // retry if at last build
        // mark the current stage as success
        // and try again via on demand node
        echo 'Exception during SPOT run ' + ex.toString() + ' retry on-demand'
        currentBuild.result = 'SUCCESS'
        run_tests('GPU')
      } else {
        echo 'Exception during SPOT run ' + ex.toString() + ' exit since it is not last build'
        throw ex
      }
    }
  }
}

def run_lint(node_type) {
  echo 'Begin running lint on node_type ' + node_type
  if (!skip_ci) {
    node(node_type) {
      ws("workspace/exec_${env.EXECUTOR_NUMBER}/flashinfer/lint") {
        init_git()
        docker_init(ci_image)
        timeout(time: max_time, unit: 'MINUTES') {
          withEnv([
            'PLATFORM=gpu',
            'TEST_STEP_NAME=lint'], {
            ci_setup(ci_image)
            sh (
              script: "${docker_run} ${ci_image} ./tests/task_lint.sh",
              label: 'Run lint',
            )
          })
        }
      }
    }
    echo 'End running lint on node_type ' + node_type
  } else {
    Utils.markStageSkippedForConditional('LINT')
  }
}

def lint() {
  stage('Lint') {
    try {
      run_lint('CPU-SPOT')
    } catch (Throwable ex) {
      if (is_last_build()) {
        echo 'Exception during SPOT run ' + ex.toString() + ' retry on-demand'
        currentBuild.result = 'SUCCESS'
        run_lint('CPU')
      } else {
        echo 'Exception during SPOT run ' + ex.toString() + ' exit since it is not last build'
        throw ex
      }
    }
  }
}

def run_unittest(node_type) {
  echo 'Begin running unittest on node_type ' + node_type
  if (!skip_ci && is_docs_only_build != 1) {
    node(node_type) {
      ws("workspace/exec_${env.EXECUTOR_NUMBER}/flashinfer/unittest") {
        init_git()
        docker_init(ci_image)
        timeout(time: max_time, unit: 'MINUTES') {
          withEnv([
            'PLATFORM=gpu',
            'TEST_STEP_NAME=unittest'
          ], {
            ci_setup(ci_image)
            sh (
              script: "${docker_run} ${ci_image} ./tests/task_unittests.sh",
              label: 'Run unit tests',
            )
          })
        }
        try {
          junit 'build/pytest-results/*.xml'
        } catch (Exception e) {
          echo 'Exception during JUnit upload: ' + e.toString()
        }
      }
    }
    echo 'End running unittest on node_type ' + node_type
  } else {
    Utils.markStageSkippedForConditional('UNITTEST')
  }
}

def unittest() {
  stage('Unit Test') {
    try {
      run_unittest('GPU-SPOT')
    } catch (Throwable ex) {
      if (is_last_build()) {
        echo 'Exception during SPOT run ' + ex.toString() + ' retry on-demand'
        currentBuild.result = 'SUCCESS'
        run_unittest('GPU')
      } else {
        echo 'Exception during SPOT run ' + ex.toString() + ' exit since it is not last build'
        throw ex
      }
    }
  }
}

build()
lint()
unittest()
