#!/usr/bin/env python3
"""
FlashInfer Code Owners Analyzer

This script analyzes git history to determine code owners for each module
in the flashinfer repository based on commit frequency and recency.
"""

import os
import subprocess
import json
import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Set, DefaultDict
import re
import urllib.request
import urllib.error
import time


class CodeOwnersAnalyzer:
    def __init__(
        self,
        repo_path: str = ".",
        min_commits: int = 2,
        days_back: int = 365,
        exclude_patterns: Optional[List[str]] = None,
        github_token: Optional[str] = None,
        use_api: bool = True,
        allowed_users: Optional[List[str]] = None,
    ):
        """
        Initialize the code owners analyzer.

        Args:
            repo_path: Path to the git repository
            min_commits: Minimum commits required to be considered an owner
            days_back: How many days back to analyze (default: 1 year)
            exclude_patterns: List of path patterns to exclude from analysis
            github_token: Optional GitHub API token for higher rate limits
            use_api: Whether to use GitHub API for email lookups (default: True)
            allowed_users: Optional list of GitHub usernames to include (filters out others)
        """
        self.repo_path = Path(repo_path).resolve()
        self.min_commits = min_commits
        self.days_back = days_back
        self.module_owners: DefaultDict[str, DefaultDict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.module_files: DefaultDict[str, Set[str]] = defaultdict(set)
        self.email_to_github: Dict[
            str, str
        ] = {}  # Cache for email to GitHub username mappings
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")
        self.use_api = use_api
        self.api_call_count = 0
        self.last_api_call_time: float = 0
        # Convert allowed users to lowercase for case-insensitive comparison
        self.allowed_users = (
            set(u.lower() for u in allowed_users) if allowed_users else None
        )

        # Default exclude patterns
        self.exclude_patterns = [
            "3rdparty/",
            ".git/",
            "__pycache__/",
            "*.pyc",
            ".pytest_cache/",
            ".tox/",
            "dist/",
            "build/",
            "*.egg-info/",
            "venv/",
            "env/",
            ".venv/",
        ]

        # Add user-provided exclude patterns
        if exclude_patterns:
            self.exclude_patterns.extend(exclude_patterns)

        # Validate repo path
        if not self.repo_path.exists():
            raise ValueError(f"Repository path does not exist: {self.repo_path}")

        if not (self.repo_path / ".git").exists():
            raise ValueError(f"Not a git repository: {self.repo_path}")

    def extract_github_username_from_email(self, email: str) -> Optional[str]:
        """
        Extract GitHub username from email address.

        Common patterns:
        - username@users.noreply.github.com
        - 12345+username@users.noreply.github.com
        - username@github.com
        - For other emails, try to use the local part as a potential username
        """
        email = email.strip().lower()

        # Check if it's already in our cache
        if email in self.email_to_github:
            return self.email_to_github[email]

        username = None

        # GitHub noreply email patterns
        if "users.noreply.github.com" in email:
            # Pattern: username@users.noreply.github.com
            match = re.match(r"^([^@+]+)@users\.noreply\.github\.com$", email)
            if match:
                username = match.group(1)
            else:
                # Pattern: 12345+username@users.noreply.github.com
                match = re.match(r"^\d+\+([^@]+)@users\.noreply\.github\.com$", email)
                if match:
                    username = match.group(1)

        # GitHub.com email
        elif "@github.com" in email:
            match = re.match(r"^([^@]+)@github\.com$", email)
            if match:
                username = match.group(1)

        # For other emails, try multiple lookup methods
        else:
            # First, try GitHub API lookup if enabled
            if self.use_api:
                username = self.lookup_github_username_via_api(email)
            else:
                username = None

            # If API lookup fails or is disabled, try to get from commit history
            if not username:
                username = self.lookup_github_username_from_commits(email)

            # If still not found, use local part of email as a last resort fallback
            # but don't return it as a username (return None instead)
            if not username:
                # We don't want to guess usernames from email local parts
                # as they're often incorrect
                username = None

        # Cache the result (including None to avoid repeated failed lookups)
        self.email_to_github[email] = username

        return username

    def lookup_github_username_via_api(self, email: str) -> Optional[str]:
        """
        Look up GitHub username via GitHub API search.

        Args:
            email: Email address to search for

        Returns:
            GitHub username if found, None otherwise
        """
        # Rate limiting based on GitHub API limits:
        # - Authenticated: 5000 requests/hour = 1.39 req/sec
        # - Unauthenticated: 60 requests/hour = 1 req/60sec
        current_time = time.time()

        if self.api_call_count > 0:
            time_since_last = current_time - self.last_api_call_time

            if self.github_token:
                # With token: 5000/hour = 0.72 seconds between calls (with buffer)
                min_delay = 0.8
            else:
                # Without token: 60/hour = 60 seconds between calls
                min_delay = 60.1

            if time_since_last < min_delay:
                time.sleep(min_delay - time_since_last)

        retry_count = 0
        max_retries = 3
        base_delay = 1

        while retry_count < max_retries:
            try:
                # Search for users by email
                search_url = f"https://api.github.com/search/users?q={email}+in:email"

                req = urllib.request.Request(search_url)
                req.add_header("Accept", "application/vnd.github.v3+json")
                req.add_header("User-Agent", "flashinfer-codeowner-analyzer")

                if self.github_token:
                    req.add_header("Authorization", f"Bearer {self.github_token}")

                self.api_call_count += 1
                self.last_api_call_time = time.time()

                with urllib.request.urlopen(req, timeout=10) as response:
                    data = json.loads(response.read().decode())

                    if data.get("total_count", 0) > 0 and "items" in data:
                        # Return the first matching user's login
                        login = data["items"][0].get("login")
                        result = login if isinstance(login, str) else None
                        # Cache the API result
                        self.email_to_github[email] = result
                        return result

                    # Cache the failed lookup to avoid retrying
                    self.email_to_github[email] = None
                    return None

            except urllib.error.HTTPError as e:
                if e.code == 403:
                    # Rate limit exceeded - implement exponential backoff
                    retry_delay = base_delay * (2**retry_count)
                    if retry_count < max_retries - 1:
                        print(
                            f"GitHub API rate limit hit, retrying in {retry_delay}s... (attempt {retry_count + 1}/{max_retries})"
                        )
                        time.sleep(retry_delay)
                        retry_count += 1
                        continue
                    else:
                        print(
                            f"Warning: GitHub API rate limit exceeded for email lookup: {email}"
                        )
                        # Cache the failed lookup to avoid retrying
                        self.email_to_github[email] = None
                        return None
                elif e.code == 401:
                    print(
                        "Warning: GitHub API authentication failed. Check your token."
                    )
                    # Cache the failed lookup to avoid retrying
                    self.email_to_github[email] = None
                    return None
                else:
                    # Other HTTP errors - don't retry but report them
                    print(
                        f"Warning: GitHub API HTTP error for {email}: {e.code} {e.reason}"
                    )
                    # Cache the failed lookup to avoid retrying
                    self.email_to_github[email] = None
                    return None
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                # Network or parsing errors - retry with backoff
                retry_delay = base_delay * (2**retry_count)
                retry_count += 1
                if retry_count < max_retries:
                    print(
                        f"GitHub API network error for {email}, retrying in {retry_delay}s... (attempt {retry_count}/{max_retries}): {type(e).__name__}"
                    )
                    time.sleep(retry_delay)
                    continue
                # Cache the failed lookup after all retries exhausted
                print(
                    f"Warning: GitHub API lookup failed for {email} after {max_retries} retries: {type(e).__name__}: {e}"
                )
                self.email_to_github[email] = None
                return None
            except Exception as e:
                # Any other errors - don't retry but report them
                print(
                    f"Warning: Unexpected error during GitHub API lookup for {email}: {type(e).__name__}: {e}"
                )
                # Cache the failed lookup to avoid retrying
                self.email_to_github[email] = None
                return None

        # Cache the failed lookup after all retries exhausted
        self.email_to_github[email] = None
        return None

    def lookup_github_username_from_commits(self, email: str) -> Optional[str]:
        """
        Try to find GitHub username from commit metadata.

        This looks for commits by this email that might have been made via GitHub
        which often includes the username in the commit message or author field.
        """
        # Look for recent commits by this author
        command = [
            "git",
            "log",
            "--author",
            email,
            "--format=%an|%cn|%s",  # author name, committer name, subject
            "--max-count=10",
        ]
        output = self.run_git_command(command)

        if not output:
            return None

        # Check if any commits were made via GitHub (often have specific patterns)
        for line in output.split("\n"):
            if line.strip():
                parts = line.split("|")
                if len(parts) >= 3:
                    # Check commit message for GitHub PR patterns
                    subject = parts[2]
                    # Pattern: "Merge pull request #123 from username/branch"
                    match = re.search(r"from ([^/\s]+)/", subject)
                    if match:
                        username = match.group(1)
                        # Cache the result
                        self.email_to_github[email] = username
                        return username

                    # Pattern: "Co-authored-by: Name <email>"
                    match = re.search(r"Co-authored-by:.*<([^>]+)>", subject)
                    if match and "users.noreply.github.com" in match.group(1):
                        username = self.extract_github_username_from_email(
                            match.group(1)
                        )
                        if username:
                            # Cache the result
                            self.email_to_github[email] = username
                            return username

        # Cache the failed lookup
        self.email_to_github[email] = None
        return None

    def should_include_contributor(self, author_string: str) -> bool:
        """
        Check if a contributor should be included based on allowed users filter.

        Args:
            author_string: Author string in format "Name <email>"

        Returns:
            True if contributor should be included, False otherwise
        """
        # If no filter is set, include everyone
        if self.allowed_users is None:
            return True

        # Get GitHub username for this author
        github_username = self.get_github_username(author_string)

        # If we can't determine the GitHub username, exclude by default when filter is active
        if not github_username:
            return False

        # Check if username is in allowed list (case-insensitive)
        return github_username.lower() in self.allowed_users

    def get_github_username(self, author_string: str) -> Optional[str]:
        """
        Extract GitHub username from author string format: "Name <email>".
        """
        # Extract email from author string
        match = re.search(r"<([^>]+)>", author_string)
        if match:
            email = match.group(1)
            return self.extract_github_username_from_email(email)
        return None

    def should_exclude(self, path: str) -> bool:
        """Check if a path should be excluded based on patterns."""
        path_obj = Path(path)

        for pattern in self.exclude_patterns:
            # Check for directory patterns
            if pattern.endswith("/"):
                if any(part == pattern.rstrip("/") for part in path_obj.parts):
                    return True
                if path.startswith(pattern) or path.startswith("./" + pattern):
                    return True
            # Check for glob patterns
            elif "*" in pattern:
                if path_obj.match(pattern):
                    return True
            # Check for exact matches or substring matches
            else:
                if pattern in path:
                    return True

        return False

    def run_git_command(self, command: List[str]) -> str:
        """Run a git command and return the output."""
        try:
            result = subprocess.run(
                command, cwd=self.repo_path, capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Error running git command: {' '.join(command)}")
            print(f"Error: {e.stderr}")
            return ""

    def get_modules(self) -> List[str]:
        """Identify modules in the flashinfer project."""
        modules = set()

        # Get all tracked files from git
        all_files = self.run_git_command(["git", "ls-files"]).split("\n")

        for file_path in all_files:
            if not file_path or self.should_exclude(file_path):
                continue

            # Skip test directories unless specifically analyzing tests
            if "test" in file_path.lower() and not file_path.startswith("./tests/"):
                continue

            # Get directory of the file
            dir_path = os.path.dirname(file_path)

            # Skip root directory
            if dir_path in [".", "./", ""]:
                continue

            # Check if it's a relevant code file
            file_ext = Path(file_path).suffix
            relevant_extensions = {
                ".py",
                ".cpp",
                ".cc",
                ".c",
                ".cu",
                ".cuh",
                ".h",
                ".hpp",
                ".hh",
                ".go",
                ".rs",
                ".java",
                ".js",
                ".ts",
                ".jsx",
                ".tsx",
            }

            if file_ext in relevant_extensions:
                # Add the directory and all parent directories as modules
                path_parts = Path(dir_path).parts
                for i in range(1, len(path_parts) + 1):
                    module = "/".join(path_parts[:i])
                    if not self.should_exclude(module):
                        modules.add(module)

        return sorted(list(modules))

    def get_file_commits(self, file_path: str, since_date: str) -> List[Dict[str, str]]:
        """Get commit information for a specific file."""
        command = [
            "git",
            "log",
            f"--since={since_date}",
            "--format=%H|%an|%ae|%ad",
            "--date=iso",
            "--",
            file_path,
        ]
        output = self.run_git_command(command)

        commits = []
        for line in output.split("\n"):
            if line.strip():
                parts = line.split("|")
                if len(parts) >= 4:
                    commits.append(
                        {
                            "hash": parts[0],
                            "author_name": parts[1],
                            "author_email": parts[2],
                            "date": parts[3],
                        }
                    )
        return commits

    def get_files_in_module(self, module_path: str) -> List[str]:
        """Get all tracked files in a module."""
        command = ["git", "ls-files", module_path]
        output = self.run_git_command(command)

        files = []
        for line in output.split("\n"):
            if line.strip() and not self.should_exclude(line.strip()):
                files.append(line.strip())
        return files

    def analyze_module_ownership(
        self, module_path: str
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Analyze ownership for a specific module."""
        since_date = (datetime.now() - timedelta(days=self.days_back)).strftime(
            "%Y-%m-%d"
        )
        files = self.get_files_in_module(module_path)

        module_contributors: DefaultDict[str, Dict[str, Any]] = defaultdict(
            lambda: {"commit_hashes": set(), "files": set(), "last_commit": None}
        )

        for file_path in files:
            commits = self.get_file_commits(file_path, since_date)

            for commit in commits:
                author_key = f"{commit['author_name']} <{commit['author_email']}>"
                # Use commit hash to ensure each commit is only counted once
                module_contributors[author_key]["commit_hashes"].add(commit["hash"])
                module_contributors[author_key]["files"].add(file_path)

                # Track most recent commit - improved date parsing
                try:
                    # Parse ISO format date with timezone
                    date_str = commit["date"]
                    # Remove timezone for parsing (handle both +0000 and -0700 formats)
                    if " +" in date_str or " -" in date_str:
                        date_str = date_str.split(" ")[0] + " " + date_str.split(" ")[1]
                    commit_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")

                    last_commit = module_contributors[author_key]["last_commit"]
                    if last_commit is None or (
                        isinstance(last_commit, datetime) and commit_date > last_commit
                    ):
                        module_contributors[author_key]["last_commit"] = commit_date
                except (ValueError, IndexError) as e:
                    # If date parsing fails, skip updating last_commit
                    print(
                        f"Warning: Could not parse date '{commit.get('date', '')}': {e}"
                    )
                    continue

        # Filter contributors based on minimum commits and allowed users
        # Convert commit_hashes to commits count for compatibility
        for data in module_contributors.values():
            commit_hashes = data["commit_hashes"]
            if isinstance(commit_hashes, set):
                data["commits"] = len(commit_hashes)

        # Apply filters: minimum commits and allowed users
        qualified_contributors = {}
        for author, data in module_contributors.items():
            # Check minimum commits
            commits = data.get("commits", 0)
            if not isinstance(commits, int) or commits < self.min_commits:
                continue

            # Check if contributor is in allowed users list
            if not self.should_include_contributor(author):
                continue

            qualified_contributors[author] = data

        return qualified_contributors, files

    def calculate_ownership_score(
        self, contributor_data: Dict[str, Any], total_commits: int, total_files: int
    ) -> float:
        """Calculate ownership score based on commits, file coverage, and recency."""
        commits = contributor_data["commits"]
        if not isinstance(commits, int):
            commits = 0

        files = contributor_data["files"]
        if isinstance(files, set):
            files_touched = len(files)
        else:
            files_touched = 0

        last_commit = contributor_data["last_commit"]

        # Base score from commit frequency
        commit_score = float(commits) / total_commits if total_commits > 0 else 0.0

        # File coverage score
        file_score = float(files_touched) / total_files if total_files > 0 else 0.0

        # Recency score (higher for more recent commits)
        if last_commit and isinstance(last_commit, datetime):
            days_since = (datetime.now() - last_commit).days
            recency_score = max(0.0, 1.0 - (days_since / self.days_back))
        else:
            recency_score = 0.0

        # Weighted combination
        ownership_score = commit_score * 0.5 + file_score * 0.3 + recency_score * 0.2

        return float(ownership_score)

    def analyze_all_modules(self, verbose: bool = True) -> Dict[str, Any]:
        """Analyze ownership for all modules."""
        modules = self.get_modules()
        results = {}

        if verbose:
            print(f"\nFound {len(modules)} modules to analyze (excluding 3rdparty/)...")
            print("=" * 50)

        total_modules = len(modules)
        for idx, module in enumerate(modules, 1):
            if verbose:
                progress = f"[{idx}/{total_modules}]"
                print(f"{progress} Analyzing module: {module}", end="\r")
            contributors, files = self.analyze_module_ownership(module)

            if not contributors:
                results[module] = {"owners": [], "files": files, "total_commits": 0}
                continue

            total_commits = sum(data["commits"] for data in contributors.values())
            total_files = len(files)

            # Calculate ownership scores
            scored_contributors = []
            for author, data in contributors.items():
                score = self.calculate_ownership_score(data, total_commits, total_files)
                github_username = self.get_github_username(author)
                scored_contributors.append(
                    {
                        "author": author,
                        "github_username": github_username,
                        "commits": data["commits"],
                        "files_touched": len(data["files"]),
                        "last_commit": data["last_commit"].isoformat()
                        if data["last_commit"]
                        else None,
                        "ownership_score": score,
                    }
                )

            # Sort by ownership score
            scored_contributors.sort(key=lambda x: x["ownership_score"], reverse=True)

            results[module] = {
                "owners": scored_contributors,
                "files": files,
                "total_commits": total_commits,
            }

        if verbose:
            print(" " * 80, end="\r")  # Clear the progress line
            print(f"✓ Analysis complete for {len(results)} modules")

        return results

    def generate_codeowners_file(
        self, results: Dict[str, Any], output_file: str = "CODEOWNERS"
    ) -> None:
        """Generate a CODEOWNERS file based on analysis."""
        with open(output_file, "w") as f:
            f.write("# Code Owners File\n")
            f.write("# Generated automatically from git history analysis\n")
            f.write(f"# Analysis period: {self.days_back} days\n")
            f.write(f"# Minimum commits threshold: {self.min_commits}\n\n")

            for module, data in results.items():
                if data["owners"]:
                    # Take top 3 owners or those with ownership score > 0.1
                    top_owners = [
                        owner
                        for owner in data["owners"][:3]
                        if owner["ownership_score"] > 0.1
                    ]

                    if top_owners:
                        # Extract GitHub usernames from author strings
                        github_usernames = []
                        for owner in top_owners:
                            github_username = self.get_github_username(owner["author"])
                            if github_username:
                                github_usernames.append(f"@{github_username}")
                            else:
                                # Fallback to email if no GitHub username found
                                email = owner["author"].split("<")[1].rstrip(">")
                                github_usernames.append(email)

                        if github_usernames:
                            owners_list = " ".join(github_usernames)
                            f.write(f"{module}/ {owners_list}\n")

    def print_detailed_report(self, results: Dict[str, Any]) -> None:
        """Print a detailed ownership report."""
        print("\n" + "=" * 80)
        print("FLASHINFER CODE OWNERSHIP ANALYSIS REPORT")
        print("=" * 80)

        for module, data in results.items():
            print(f"\n📁 Module: {module}")
            print(f"   Files: {len(data['files'])}")
            print(
                f"   Total commits (last {self.days_back} days): {data['total_commits']}"
            )

            if data["owners"]:
                print("   👥 Code Owners (by ownership score):")
                for i, owner in enumerate(data["owners"][:5], 1):
                    last_commit = owner["last_commit"]
                    if last_commit:
                        last_commit = datetime.fromisoformat(last_commit).strftime(
                            "%Y-%m-%d"
                        )
                    else:
                        last_commit = "N/A"

                    # Get GitHub username
                    github_username = self.get_github_username(owner["author"])
                    github_info = f" [@{github_username}]" if github_username else ""

                    print(f"      {i}. {owner['author']}{github_info}")
                    print(
                        f"         Commits: {owner['commits']}, "
                        f"Files: {owner['files_touched']}, "
                        f"Score: {owner['ownership_score']:.3f}, "
                        f"Last: {last_commit}"
                    )
            else:
                print("   ⚠️  No active contributors found (below minimum threshold)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze code ownership in flashinfer repository",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Basic analysis
  %(prog)s --exclude vendor/ deps/  # Exclude additional directories
  %(prog)s --days-back 180          # Analyze last 6 months
  %(prog)s --json-output owners.json # Export detailed JSON
  %(prog)s --github-token TOKEN     # Use GitHub API for email lookups
  GITHUB_TOKEN=TOKEN %(prog)s       # Or set via environment variable
  %(prog)s --allowed-users user1 user2 # Only include specific GitHub users
  %(prog)s --allowed-users-file team.txt # Load allowed users from file
        """,
    )
    parser.add_argument("--repo-path", default=".", help="Path to git repository")
    parser.add_argument(
        "--min-commits",
        type=int,
        default=1,
        help="Minimum commits to be considered owner",
    )
    parser.add_argument(
        "--days-back", type=int, default=365, help="Days to analyze back in history"
    )
    parser.add_argument(
        "--output", default="CODEOWNERS", help="Output file for CODEOWNERS"
    )
    parser.add_argument("--json-output", help="Output detailed results as JSON")
    parser.add_argument(
        "--quiet", action="store_true", help="Only output final results"
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Additional path patterns to exclude (e.g., vendor/ deps/)",
    )
    parser.add_argument(
        "--github-token",
        help="GitHub API token for email lookups (or set GITHUB_TOKEN env var)",
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Disable GitHub API lookups for faster processing",
    )
    parser.add_argument(
        "--allowed-users",
        nargs="*",
        help="Only include these GitHub users in analysis (e.g., user1 user2)",
    )
    parser.add_argument(
        "--allowed-users-file",
        help="File containing allowed GitHub usernames, one per line",
    )

    args = parser.parse_args()

    # Process allowed users list
    allowed_users = []
    if args.allowed_users:
        allowed_users.extend(args.allowed_users)

    if args.allowed_users_file:
        try:
            with open(args.allowed_users_file, "r") as f:
                # Read usernames from file, one per line, skip empty lines and comments
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        allowed_users.append(line)
        except FileNotFoundError:
            print(
                f"Error: Allowed users file not found: {args.allowed_users_file}",
                file=sys.stderr,
            )
            return 1
        except Exception as e:
            print(f"Error reading allowed users file: {e}", file=sys.stderr)
            return 1

    # Remove duplicates
    allowed_users = list(set(allowed_users)) if allowed_users else None

    try:
        analyzer = CodeOwnersAnalyzer(
            repo_path=args.repo_path,
            min_commits=args.min_commits,
            days_back=args.days_back,
            exclude_patterns=args.exclude,
            github_token=args.github_token,
            use_api=not args.no_api,
            allowed_users=allowed_users,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print("\n🔍 Starting FlashInfer Code Ownership Analysis")
        print("=" * 50)
        print(f"📁 Repository: {analyzer.repo_path}")
        print(f"📅 Analysis period: {args.days_back} days")
        print(f"📊 Minimum commits threshold: {args.min_commits}")
        print(
            f"🚫 Excluding: {', '.join(analyzer.exclude_patterns[:5])}{'...' if len(analyzer.exclude_patterns) > 5 else ''}"
        )
        if allowed_users:
            print(
                f"👥 Filtering to {len(allowed_users)} allowed users: {', '.join(allowed_users[:10])}{'...' if len(allowed_users) > 10 else ''}"
            )

    results = analyzer.analyze_all_modules(verbose=not args.quiet)

    # Generate CODEOWNERS file
    analyzer.generate_codeowners_file(results, args.output)

    # Save JSON output if requested
    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {args.json_output}")

    # Print detailed report
    if not args.quiet:
        analyzer.print_detailed_report(results)

    print(f"\n✅ CODEOWNERS file generated: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
