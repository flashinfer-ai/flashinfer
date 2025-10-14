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
from typing import List, Dict, Optional, Tuple, Any, DefaultDict
import re


class CodeOwnersAnalyzer:
    def __init__(
        self,
        repo_path: str = ".",
        min_commits: int = 2,
        days_back: int = 365,
        exclude_patterns: Optional[List[str]] = None,
        allowed_users: Optional[List[str]] = None,
        max_depth: int = 3,
        top_n_owners: int = 3,
    ):
        """
        Initialize the code owners analyzer.

        Args:
            repo_path: Path to the git repository
            min_commits: Minimum commits required to be considered an owner
            days_back: How many days back to analyze (default: 1 year)
            exclude_patterns: List of path patterns to exclude from analysis
            allowed_users: Optional list of GitHub usernames to include (filters out others)
            max_depth: Maximum directory depth for module detection (default: 3)
            top_n_owners: Number of top owners to include in CODEOWNERS file (default: 3)
        """
        self.repo_path = Path(repo_path).resolve()
        self.min_commits = min_commits
        self.days_back = days_back
        self.max_depth = max_depth
        self.top_n_owners = top_n_owners
        self.email_to_github: Dict[
            str, str
        ] = {}  # Cache for email to GitHub username mappings
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

        # Check if gh CLI is available
        try:
            subprocess.run(
                ["gh", "--version"], capture_output=True, check=True, timeout=5
            )
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ) as e:
            raise ValueError(
                "GitHub CLI (gh) is not installed or not available in PATH.\n"
                "Please install it from: https://cli.github.com/\n"
                "Or use package manager: brew install gh / apt install gh / etc."
            ) from e

    def extract_github_username_from_email(self, email: str) -> Optional[str]:
        """
        Extract GitHub username from email address.

        For GitHub noreply emails, extract directly from email pattern.
        For all other emails, use GitHub CLI to lookup the username.
        """
        email = email.strip().lower()

        # Check if it's already in our cache
        if email in self.email_to_github:
            return self.email_to_github[email]

        username = None

        # GitHub noreply email patterns - can extract directly
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
        else:
            # For all other emails, use GitHub CLI to lookup
            username = self.lookup_github_username_via_gh_cli(email)

        # Cache the result (including None to avoid repeated failed lookups)
        self.email_to_github[email] = username

        return username

    def lookup_github_username_via_gh_cli(self, email: str) -> Optional[str]:
        """
        Look up GitHub username using the GitHub CLI tool (gh).

        This queries the GitHub repository commits to find commits by the given email
        and extracts the author's GitHub login name.

        Args:
            email: Email address to search for

        Returns:
            GitHub username if found, None otherwise
        """
        try:
            # Extract repository owner and name from git remote
            remote_url = self.run_git_command(
                ["git", "config", "--get", "remote.origin.url"]
            )

            if not remote_url:
                return None

            # Parse GitHub repo from URL (supports both HTTPS and SSH formats)
            # HTTPS: https://github.com/owner/repo.git
            # SSH: git@github.com:owner/repo.git
            repo_match = re.search(
                r"github\.com[:/]([^/]+)/([^/\s]+?)(?:\.git)?$", remote_url
            )
            if not repo_match:
                return None

            repo_owner = repo_match.group(1)
            repo_name = repo_match.group(2)
            repo_full = f"{repo_owner}/{repo_name}"

            # Use gh CLI to search for commits by this author email
            # Use author filter in URL query string
            gh_command = [
                "gh",
                "api",
                f"repos/{repo_full}/commits?author={email}&per_page=1",
                "--jq",
                ".[0].author.login // empty",
            ]

            result = subprocess.run(
                gh_command, capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0 and result.stdout.strip():
                username = result.stdout.strip()
                return username if username else None

            return None

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, Exception):
            # If gh CLI fails or is not available, return None
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
                # Limited by max_depth
                path_parts = Path(dir_path).parts
                max_parts = min(len(path_parts), self.max_depth)
                for i in range(1, max_parts + 1):
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
                    # Take top N owners or those with ownership score > 0.1
                    top_owners = [
                        owner
                        for owner in data["owners"][: self.top_n_owners]
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
        "--allowed-users",
        nargs="*",
        help="Only include these GitHub users in analysis (e.g., user1 user2)",
    )
    parser.add_argument(
        "--allowed-users-file",
        help="File containing allowed GitHub usernames, one per line",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Maximum directory depth for module detection (default: 3)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top owners to include in CODEOWNERS file (default: 3)",
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
            allowed_users=allowed_users,
            max_depth=args.depth,
            top_n_owners=args.top_n,
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
