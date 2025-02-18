pipeline {
    agent any
    
    stages {
        stage('Checkout') {
            steps {
                git 'https://your-repo-url.git'
            }
        }
        
        stage('Build') {
            steps {
                echo 'Building...'
                // Add build commands here
            }
        }
        
        stage('Test') {
            steps {
                echo 'Running tests...'
                // Add test commands here
            }
        }
        
        stage('Deploy') {
            steps {
                echo 'Deploying...'
                // Add deployment commands here
            }
        }
    }
}

