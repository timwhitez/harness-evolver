"""Tests for tool registry, registration, and dispatch."""

import shlex
from pathlib import Path

import pytest

from harness.tools.registry import ToolRegistry
from harness.tools.shell import (
    ShellTool,
    background_package_command_reason,
    broad_proc_scan_command_reason,
    broad_root_find_command_reason,
    external_agent_command_reason,
    heavy_graphics_runtime_install_reason,
    heavy_scientific_dependency_install_reason,
    large_graphics_runtime_install_plan_reason,
    large_package_install_plan_reason,
    large_toolchain_install_command_reason,
    large_toolchain_install_plan_reason,
    manual_deb_dependency_chase_reason,
    manual_dependency_download_reason,
    package_manager_timeout_cap,
    scripted_package_manager_command_reason,
    shell_semantic_failure_kind,
)
from harness.tools.file_read import FileReadTool
from harness.tools.file_edit import FileEditTool
from harness.tools.file_write import FileWriteTool
from harness.tools.search import GrepTool, GlobTool
from harness.tools.verify import VerifyTool
from harness.tools.verify import verify_semantic_failure_kind


def _assert_policy_guard_is_non_terminal(metadata):
    assert metadata["policy_guard_stop_condition"] is False
    assert metadata["operation_guard_stop_condition"] is False
    assert metadata["loop_stop_condition"] is False


def test_tools_package_exports_core_and_harness_tools():
    import harness.tools as tools

    assert tools.ShellTool is ShellTool
    assert "ShellTool" in tools.__all__
    assert "TodoWriteTool" in tools.__all__
    assert "ToolFailureTracker" in tools.__all__


def test_tool_policy_guards_use_non_terminal_metadata_helper():
    repo = Path(__file__).resolve().parents[1]
    source_text = "\n".join(
        (repo / path).read_text()
        for path in [
            "harness/tools/shell.py",
            "harness/tools/file_read.py",
            "harness/tools/file_write.py",
            "harness/tools/file_edit.py",
            "harness/tools/search.py",
            "bench/harbor_adapter.py",
        ]
    )

    assert 'metadata={"blocked_by"' not in source_text
    assert "policy_guard_metadata(" in source_text


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = ShellTool()
        registry.register(tool)
        assert registry.get("bash") is tool

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register(ShellTool())
        registry.register(FileReadTool())
        assert "bash" in registry.list_tools()
        assert "read" in registry.list_tools()

    def test_unregister(self):
        registry = ToolRegistry()
        registry.register(ShellTool())
        registry.unregister("bash")
        assert registry.get("bash") is None

    def test_unknown_tool(self):
        registry = ToolRegistry()
        result = registry.execute("nonexistent")
        assert result.success is False
        assert "Unknown tool" in result.error

    def test_get_schemas(self):
        registry = ToolRegistry()
        registry.register(ShellTool())
        schemas = registry.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "bash"

    def test_validate_empty(self):
        registry = ToolRegistry()
        errors = registry.validate_all()
        assert errors == []


class TestShellTool:
    def test_basic_execution(self):
        tool = ShellTool()
        result = tool.execute("echo hello")
        assert result.success is True
        assert "hello" in result.output

    def test_failed_command(self):
        tool = ShellTool()
        result = tool.execute("exit 1")
        assert result.success is False
        assert "exit code" in result.error

    def test_pipeline_failure_uses_bash_pipefail(self):
        tool = ShellTool()
        result = tool.execute("python3 -c 'import sys; sys.exit(3)' | cat")
        assert result.success is False
        assert result.metadata["exit_code"] == 3
        assert "exit code: 3" in result.error

    def test_package_manager_semantic_failure_in_successful_pipeline(self):
        tool = ShellTool()
        result = tool.execute(
            "python3 -c \"print('ERROR: No matching distribution found for definitely_missing_pkg')\" | cat"
        )
        assert result.success is False
        assert result.metadata["semantic_failure_detected"] is True
        assert result.metadata["semantic_failure_kind"] == "package_manager_failure"
        assert "package manager output indicates failure" in result.error

    def test_package_manager_pep668_output_is_semantic_failure(self):
        tool = ShellTool()
        result = tool.execute(
            "python3 -c \"print('error: externally-managed-environment')\" | cat"
        )

        assert result.success is False
        assert result.metadata["semantic_failure_detected"] is True
        assert result.metadata["semantic_failure_kind"] == "package_manager_failure"

    @pytest.mark.parametrize(
        "package_output",
        [
            "dpkg: dependency problems prevent configuration of libstdc++-12-dev:amd64",
            "ERROR: failed to lock directory '/usr/local/lib/R/site-library' for modifying",
            "E: Could not get lock /var/lib/dpkg/lock-frontend. It is held by process 230",
        ],
    )
    def test_package_manager_dpkg_and_r_lock_outputs_are_semantic_failures(
        self, package_output
    ):
        tool = ShellTool()
        result = tool.execute(f"printf '%s\n' {package_output!r} | head -20")

        assert result.success is False
        assert result.metadata["semantic_failure_detected"] is True
        assert result.metadata["semantic_failure_kind"] == "package_manager_failure"

    @pytest.mark.parametrize(
        "package_output",
        [
            "Could not fetch URL https://pypi.org/simple/setuptools/: Max retries exceeded",
            "pip._vendor.pyproject_hooks._impl.BackendUnavailable: Cannot import 'setuptools.build_meta'",
            "no such option: --break-system-packages",
            "Warning: unable to access index for repository https://cloud.r-project.org/src/contrib",
        ],
    )
    def test_package_manager_history_outputs_are_semantic_failures(self, package_output):
        tool = ShellTool()
        result = tool.execute(f"printf '%s\\n' {package_output!r} | tail -5")

        assert result.success is False
        assert result.metadata["semantic_failure_detected"] is True
        assert result.metadata["semantic_failure_kind"] == "package_manager_failure"

    def test_network_probe_missing_tool_hidden_by_pipeline_is_semantic_failure(self):
        kind = shell_semantic_failure_kind(
            "bash: line 1: ping: command not found",
            command="ping -c 1 -W 2 deb.debian.org 2>&1 | head -5",
            returncode=0,
        )

        assert kind == "network_probe_tool_missing"

    @pytest.mark.parametrize(
        "command, output",
        [
            ("curl --head --connect-timeout 5 https://example.com | head -5", "curl: command not found"),
            ("wget --spider --timeout=5 https://example.com 2>&1 | head -5", "sh: 1: wget: not found"),
            ("nc -z example.com 443 2>&1 | head -5", "nc: command not found"),
        ],
    )
    def test_network_probe_missing_tool_variants_are_semantic_failures(
        self, command, output
    ):
        assert (
            shell_semantic_failure_kind(output, command=command, returncode=0)
            == "network_probe_tool_missing"
        )

    def test_non_probe_missing_tool_output_is_not_semantic_failure(self):
        kind = shell_semantic_failure_kind(
            "bash: line 1: render_model: command not found",
            command="render_model input.obj > output.txt 2>&1 | head -20",
            returncode=0,
        )

        assert kind is None

    def test_curl_package_download_missing_tool_is_not_network_probe_semantic_failure(self):
        kind = shell_semantic_failure_kind(
            "curl: command not found",
            command=(
                "curl --connect-timeout 5 -L "
                "https://files.pythonhosted.org/packages/pkg.tar.gz "
                "-o /tmp/pkg.tar.gz | head -5"
            ),
            returncode=0,
        )

        assert kind is None

    def test_dependency_setup_sigkill_hidden_by_pipeline_is_semantic_failure(self):
        tool = ShellTool()
        result = tool.execute(
            "printf '%s\\n' 'building wheel for fasttext' 'exit code: 137' | tail -2"
        )

        assert result.success is False
        assert result.metadata["exit_code"] == 0
        assert result.metadata["semantic_failure_detected"] is True
        assert result.metadata["semantic_failure_kind"] == "package_manager_failure"
        assert "package manager output indicates failure" in result.error

    def test_background_package_monitor_sigkill_is_semantic_failure(self):
        tool = ShellTool()
        result = tool.execute(
            "printf '%s\\n' 'exit code: 137' | tail -1 # tail /tmp/apt_install.log"
        )

        assert result.success is False
        assert result.metadata["exit_code"] == 0
        assert result.metadata["semantic_failure_detected"] is True
        assert result.metadata["semantic_failure_kind"] == "package_manager_failure"
        assert "package manager output indicates failure" in result.error

    def test_exit_137_log_inspection_without_setup_context_is_not_semantic_failure(self):
        tool = ShellTool()
        result = tool.execute(
            "printf '%s\\n' 'previous run ended with exit code: 137' | tail -1; echo inspected"
        )

        assert result.success is True
        assert "semantic_failure_detected" not in result.metadata

    def test_package_manager_version_check_warning_after_install_is_not_failure(self):
        tool = ShellTool()
        result = tool.execute(
            "printf '%s\\n' "
            "'Successfully installed pybind11-3.0.4' "
            "'Could not fetch URL https://pypi.org/simple/pip/: certificate verify failed'"
        )

        assert result.success is True
        assert "semantic_failure_detected" not in result.metadata

    @pytest.mark.parametrize(
        "command, package",
        [
            (
                "apt-get install -y r-cran-rstan r-cran-stanheaders r-cran-rstantools r-cran-bh",
                "r-cran-rstan",
            ),
            ("apt-get install --no-upgrade -y r-cran-rstan", "r-cran-rstan"),
            ("pip install --trusted-host pypi.org httpstan", "httpstan"),
            ("python3 -m pip install pystan httpstan", "pystan"),
            ("pip install fasttext", "fasttext"),
            ("pip install fasttext-wheel", "fasttext-wheel"),
            (
                "Rscript -e 'install.packages(\"rstan\", repos=\"https://cloud.r-project.org\")'",
                "rstan",
            ),
            ("R CMD INSTALL /tmp/rstan_2.32.7.tar.gz 2>&1 | tail -40", "rstan"),
        ],
    )
    def test_blocks_heavy_scientific_dependency_installs_before_execution(
        self, command, package
    ):
        tool = ShellTool()
        result = tool.execute(command)

        assert result.success is False
        assert result.metadata["blocked_by"] == "heavy_scientific_dependency_guard"
        _assert_policy_guard_is_non_terminal(result.metadata)
        assert package in result.error
        assert "source builds" in result.error
        assert "dependency-free" in result.error

    @pytest.mark.parametrize(
        "command",
        [
            "apt-get install -y jq",
            "pip install packaging",
            "python3 -m pip install packaging",
            "Rscript -e 'install.packages(\"jsonlite\")'",
            "printf 'pip install fasttext should not run here\n'",
        ],
    )
    def test_heavy_scientific_dependency_guard_allows_smaller_or_text_paths(
        self, command
    ):
        assert heavy_scientific_dependency_install_reason(command) is None

    @pytest.mark.parametrize(
        "command, package",
        [
            ("apt-get install -y libgl1 libgl1-mesa-dri mesa-vulkan-drivers", "libgl1"),
            ("apt install -y python3-opencv", "python3-opencv"),
            ("pip install opencv-python", "opencv-python"),
        ],
    )
    def test_blocks_heavy_graphics_runtime_installs_before_execution(
        self, command, package
    ):
        tool = ShellTool()
        result = tool.execute(command)

        assert result.success is False
        assert result.metadata["blocked_by"] == "heavy_graphics_runtime_dependency_guard"
        assert package in result.error
        assert "Mesa/OpenGL/Vulkan/X11" in result.error
        assert "dependency-light CV artifact" in result.error

    @pytest.mark.parametrize(
        "command",
        [
            "apt-get install -y jq",
            "pip install packaging",
            "printf 'apt install libgl1 should not run here\n'",
        ],
    )
    def test_heavy_graphics_runtime_guard_allows_smaller_or_text_paths(
        self, command
    ):
        assert heavy_graphics_runtime_install_reason(command) is None

    def test_staged_script_blocks_heavy_graphics_runtime_installs(self):
        from harness.tools.shell import staged_dependency_script_reason

        reason = staged_dependency_script_reason(
            "/app/setup_cv.sh",
            "#!/bin/sh\napt-get install -y libgl1 mesa-vulkan-drivers\n",
        )

        assert reason
        assert "staged script contains heavy graphics/CV runtime installs" in reason
        assert "Mesa/OpenGL/Vulkan/X11" in reason

    def test_masked_build_test_failure_is_semantic_failure(self):
        tool = ShellTool()
        result = tool.execute(
            "tmp=$(mktemp -d); "
            "printf '%s\n' "
            "'import unittest' "
            "'class T(unittest.TestCase):' "
            "'    def test_fail(self):' "
            "'        self.fail(\"boom\")' "
            "'if __name__ == \"__main__\": unittest.main()' "
            "> \"$tmp/test_fail.py\"; "
            "python3 -m unittest discover \"$tmp\" 2>&1 || true",
            timeout=5,
        )

        assert result.success is False
        assert result.metadata["exit_code"] == 0
        assert result.metadata["semantic_failure_detected"] is True
        assert result.metadata["semantic_failure_kind"] == "masked_build_test_failure"
        assert "build/test output indicates failure" in result.error
        assert "FAILED" in result.output

    def test_failure_log_inspection_is_not_semantic_failure(self):
        tool = ShellTool()
        result = tool.execute(
            "printf '%s\\n' 'make: *** [Makefile:12: all] Error 2' 'BUILD_EXIT=2' | tail -2; echo inspected",
            timeout=5,
        )

        assert result.success is True
        assert "semantic_failure_detected" not in result.metadata

    def test_heavy_ml_cv_import_failure_hidden_by_success_exit_is_semantic_failure(self):
        tool = ShellTool()
        output = (
            "/usr/local/lib/python3.11/site-packages/cv2/__init__.py: line 181\n"
            "ImportError: libGL.so.1: cannot open shared object file: No such file or directory\n"
        )
        result = tool.execute(
            f"printf '%s' {shlex.quote(output)}",
            timeout=5,
        )

        assert result.success is False
        assert result.metadata["exit_code"] == 0
        assert result.metadata["semantic_failure_detected"] is True
        assert result.metadata["semantic_failure_kind"] == "heavy_ml_cv_import_failure"
        assert "heavy ML/CV import output" in result.error
        assert "dependency-light artifact" in result.error

    def test_heavy_ml_cv_import_helper_ignores_unrelated_missing_module_log(self):
        assert shell_semantic_failure_kind(
            "ModuleNotFoundError: No module named 'example'",
            command="cat old.log",
            returncode=0,
        ) is None

    def test_numpy_eigensolver_failure_hidden_by_success_exit_is_semantic_failure(self):
        tool = ShellTool()
        output = (
            "from eigen import find_dominant_eigenvalue_and_eigenvector\n"
            "trial 0: error Cannot cast ufunc 'subtract' output from "
            "dtype('complex128') to dtype('float64') with casting rule 'same_kind'\n"
        )
        result = tool.execute(
            f"printf '%s' {shlex.quote(output)}",
            timeout=5,
        )

        assert result.success is False
        assert result.metadata["exit_code"] == 0
        assert result.metadata["semantic_failure_detected"] is True
        assert result.metadata["semantic_failure_kind"] == "numpy_eigensolver_failure"
        assert "NumPy eigensolver output" in result.error
        assert "eigen.py" in result.error

    def test_numpy_eigensolver_helper_ignores_plain_log_inspection(self):
        assert shell_semantic_failure_kind(
            "Cannot cast ufunc 'subtract' output from dtype('complex128') to dtype('float64')",
            command="cat old-eigen.log",
            returncode=0,
        ) is None

    def test_numpy_eigensolver_speed_failure_hidden_by_success_exit_is_semantic_failure(self):
        tool = ShellTool()
        output = (
            "FAILED test_outputs.py::test_speedup\n"
            "from eigen import find_dominant_eigenvalue_and_eigenvector\n"
            ">       assert dt < ref_dt, f\"{dt:.6f} seconds/call > {ref_dt:.6f} seconds/call\"\n"
            "E       AssertionError: 0.000026 seconds/call > 0.000025 seconds/call\n"
        )
        result = tool.execute(
            f"printf '%s' {shlex.quote(output)}",
            timeout=5,
        )

        assert result.success is False
        assert result.metadata["exit_code"] == 0
        assert result.metadata["semantic_failure_detected"] is True
        assert (
            result.metadata["semantic_failure_kind"]
            == "numpy_eigensolver_speed_threshold_failure"
        )
        assert "slower than the reference" in result.error
        assert "sizes 2-10" in result.error

    def test_numpy_eigensolver_speed_helper_ignores_plain_log_inspection(self):
        assert shell_semantic_failure_kind(
            "FAILED test_outputs.py::test_speedup\n"
            "E AssertionError: 0.000026 seconds/call > 0.000025 seconds/call\n",
            command="tail -50 old-eigen.log",
            returncode=0,
        ) is None

    def test_single_file_deliverable_directory_extra_file_hidden_by_success_exit(self):
        tool = ShellTool()
        output = (
            "def test_fibonacci_polyglot():\n"
            "    polyglot_files = os.listdir(\"/app/polyglot\")\n"
            ">       assert polyglot_files == [\"main.py.c\"], (\n"
            "E       AssertionError: Expected only main.py.c, found: ['cmain', 'main.py.c']\n"
        )
        result = tool.execute(f"printf '%s' {shlex.quote(output)}", timeout=5)

        assert result.success is False
        assert result.metadata["exit_code"] == 0
        assert result.metadata["semantic_failure_detected"] is True
        assert (
            result.metadata["semantic_failure_kind"]
            == "single_file_deliverable_directory_contract"
        )
        assert "single-file deliverable directory" in result.error
        assert "os.listdir exact-file-list" in result.error

    def test_single_file_deliverable_directory_missing_dir_hidden_by_success_exit(self):
        tool = ShellTool()
        output = (
            "def test_fibonacci_polyglot():\n"
            ">       polyglot_files = os.listdir(\"/app/polyglot\")\n"
            "E       FileNotFoundError: [Errno 2] No such file or directory: '/app/polyglot'\n"
        )
        result = tool.execute(f"printf '%s' {shlex.quote(output)}", timeout=5)

        assert result.success is False
        assert result.metadata["semantic_failure_kind"] == (
            "single_file_deliverable_directory_contract"
        )
        assert "create /app/polyglot early" in result.error

    def test_single_file_deliverable_directory_helper_ignores_plain_log_inspection(self):
        assert shell_semantic_failure_kind(
            "E AssertionError: Expected only main.rs, found: ['main.rs', 'cmain']\n",
            command="cat verifier.log",
            returncode=0,
        ) is None

    def test_gpt2_codegolf_size_failure_hidden_by_success_exit(self):
        tool = ShellTool()
        output = (
            "def test_gpt2_implementation():\n"
            "    gpt2_path = Path(\"/app/gpt2.c\")\n"
            ">       assert gpt2_path.stat().st_size < 5000\n"
            "E       AssertionError: File /app/gpt2.c is larger than 5000 bytes\n"
            "E       assert 5466 < 5000\n"
        )
        result = tool.execute(f"printf '%s' {shlex.quote(output)}", timeout=5)

        assert result.success is False
        assert result.metadata["exit_code"] == 0
        assert result.metadata["semantic_failure_detected"] is True
        assert result.metadata["semantic_failure_kind"] == "gpt2_codegolf_text_contract"
        assert "GPT2 codegolf text contract" in result.error
        assert "gcc -O3 /app/gpt2.c -lm" in result.error
        assert "WARRANTY OF ANY KIND, EXPRESS OR IMPLIED" in result.error

    def test_gpt2_codegolf_helper_ignores_plain_log_inspection(self):
        assert shell_semantic_failure_kind(
            "E AssertionError: File /app/gpt2.c is larger than 5000 bytes\n",
            command="tail -50 verifier.log",
            returncode=0,
        ) is None

    def test_gpt2_codegolf_helper_ignores_unrelated_size_failure(self):
        assert shell_semantic_failure_kind(
            "E AssertionError: File /app/answer.txt is larger than 5000 bytes\n",
            command="python3 -m pytest /tests/test_outputs.py",
            returncode=0,
        ) is None

    def test_structured_csv_table_row_failure_hidden_by_success_exit(self):
        tool = ShellTool()
        output = (
            "def test_summary_csv_content():\n"
            "    summary_file = Path('/app/invoices/summary.csv')\n"
            "    df = pd.read_csv(summary_file)\n"
            ">       assert len(df) == len(expected_data), 'Expected 11 rows'\n"
            "E       AssertionError: Expected 11 rows\n"
            "E       assert 10 == 11\n"
        )
        result = tool.execute(f"printf '%s' {shlex.quote(output)}", timeout=5)

        assert result.success is False
        assert result.metadata["exit_code"] == 0
        assert result.metadata["semantic_failure_detected"] is True
        assert result.metadata["semantic_failure_kind"] == "structured_csv_table_contract"
        assert "structured CSV/table contract" in result.error
        assert "pd.read_csv" in result.error
        assert "header/column order" in result.error
        assert "blank-vs-nonblank" in result.error

    def test_structured_csv_table_helper_ignores_plain_log_inspection(self):
        assert shell_semantic_failure_kind(
            "df = pd.read_csv(summary_file)\n"
            "E AssertionError: Expected 11 rows\n",
            command="cat verifier.log",
            returncode=0,
        ) is None

    def test_structured_csv_table_helper_ignores_successful_shape_inspection(self):
        assert shell_semantic_failure_kind(
            "df = pd.read_csv('/app/invoices/summary.csv')\n"
            "Columns: ['filename', 'total_amount', 'vat_amount']\n"
            "Shape: (11, 3)\n",
            command="python3 - <<'PY'\nimport pandas as pd\nprint(pd.read_csv('/app/invoices/summary.csv').shape)\nPY",
            returncode=0,
        ) is None

    def test_missing_output_artifact_failure_hidden_by_success_exit(self):
        tool = ShellTool()
        output = (
            "def test_hello_file_exists():\n"
            "    hello_path = Path(\"/app/out.txt\")\n"
            ">   assert hello_path.exists(), f\"File {hello_path} does not exist\"\n"
            "E   AssertionError: File /app/out.txt does not exist\n"
            "E    + where exists = PosixPath('/app/out.txt').exists\n"
        )
        result = tool.execute(f"printf '%s' {shlex.quote(output)}", timeout=5)

        assert result.success is False
        assert result.metadata["exit_code"] == 0
        assert result.metadata["semantic_failure_detected"] is True
        assert result.metadata["semantic_failure_kind"] == "missing_output_artifact_contract"
        assert "missing output artifact contract" in result.error
        assert "verifier-named /app artifact path" in result.error
        assert "Path(...).exists()" in result.error

    def test_missing_output_artifact_helper_uses_direct_analysis_marker(self):
        assert shell_semantic_failure_kind(
            "missing output artifact(s): /app/model.bin\n",
            command="python3 -m pytest /tests/test_outputs.py || true",
            returncode=0,
        ) == "missing_output_artifact_contract"

    def test_missing_output_artifact_helper_ignores_plain_log_inspection(self):
        assert shell_semantic_failure_kind(
            "E AssertionError: File /app/out.txt does not exist\n",
            command="tail -50 verifier.log",
            returncode=0,
        ) is None

    def test_missing_output_artifact_helper_ignores_input_file_missing(self):
        assert shell_semantic_failure_kind(
            "def test_reads_input():\n"
            "    input_path = Path('/app/input.csv')\n"
            "E   FileNotFoundError: [Errno 2] No such file or directory: '/app/input.csv'\n",
            command="python3 -m pytest /tests/test_outputs.py",
            returncode=0,
        ) is None

    def test_dna_insert_primer_pair_failure_hidden_by_success_exit(self):
        tool = ShellTool()
        output = (
            "def test_primers():\n"
            "    primers_path = Path('/app/primers.fasta')\n"
            "    assert len(lines) == 4, 'Invalid number of lines in primers.fasta.'\n"
            "    fwd_primer = lines[1].lower()\n"
            "    rev_primer = lines[3].lower()\n"
            "    primers_concat = rc(rev_primer) + fwd_primer\n"
            "    insert_start = primers_concat.find(insert)\n"
            ">   assert insert_start != -1, 'Primer must contain inserted DNA.'\n"
            "E   AssertionError: Primer must contain inserted DNA.\n"
            "Forward annealing length 0: FAIL (need 15-45)\n"
        )
        result = tool.execute(f"printf '%s' {shlex.quote(output)}", timeout=5)

        assert result.success is False
        assert result.metadata["exit_code"] == 0
        assert result.metadata["semantic_failure_detected"] is True
        assert result.metadata["semantic_failure_kind"] == "dna_insert_primer_pair_contract"
        assert "DNA insert primer-pair contract" in result.error
        assert "primers_concat = rc(rev_primer) + fwd_primer" in result.error
        assert "15..45" in result.error
        assert "Tm delta <=5" in result.error

    def test_dna_primer_helper_ignores_plain_log_inspection(self):
        assert shell_semantic_failure_kind(
            "primers_path = Path('/app/primers.fasta')\n"
            "primers_concat = rc(rev_primer) + fwd_primer\n"
            "E AssertionError: Primer must contain inserted DNA.\n",
            command="tail -50 verifier.log",
            returncode=0,
        ) is None

    def test_dna_primer_missing_file_remains_missing_artifact_contract(self):
        assert shell_semantic_failure_kind(
            "def test_primers():\n"
            "    primers_path = Path('/app/primers.fasta')\n"
            "E   FileNotFoundError: [Errno 2] No such file or directory: '/app/primers.fasta'\n",
            command="python3 -m pytest /tests/test_outputs.py",
            returncode=0,
        ) == "missing_output_artifact_contract"

    def test_dna_primer_helper_ignores_unrelated_primer_text(self):
        assert shell_semantic_failure_kind(
            "primer design notes: Tm should be close and overlap should be long\n",
            command="python3 - <<'PY'\nprint('primer design notes')\nPY",
            returncode=0,
        ) is None

    @pytest.mark.parametrize(
        "command, package",
        [
            ("apt-get install -y clang", "clang"),
            (
                "DEBIAN_FRONTEND=noninteractive apt-get install -y gcc-mipsel-linux-gnu",
                "gcc-mipsel-linux-gnu",
            ),
            ("timeout 60 apt-get install -y --no-install-recommends g++", "g++"),
            (
                "dpkg -i /var/cache/apt/archives/binutils-mipsel-linux-gnu_2.40-2cross2_amd64.deb",
                "binutils-mipsel-linux-gnu",
            ),
            (
                "dpkg -i /tmp/g++-12_12.2.0-14+deb12u1_amd64.deb",
                "g++-12",
            ),
        ],
    )
    def test_blocks_large_toolchain_installs_before_execution(self, command, package):
        tool = ShellTool()
        result = tool.execute(command)

        assert result.success is False
        assert result.metadata["blocked_by"] == "large_toolchain_install_guard"
        _assert_policy_guard_is_non_terminal(result.metadata)
        assert package in result.error
        assert "hundreds of MB" in result.error

    @pytest.mark.parametrize(
        "command",
        [
            "apt-cache search mipsel",
            "apt-get install -y jq",
            "printf 'apt-get install clang should not run here\\n'",
        ],
    )
    def test_large_toolchain_install_guard_allows_non_toolchain_commands(self, command):
        assert large_toolchain_install_command_reason(command) is None

    @pytest.mark.parametrize(
        "command, package",
        [
            (
                "dpkg -i /tmp/r-cran-rcppparallel_5.1.10-1_amd64.deb",
                "r-cran-rcppparallel",
            ),
            (
                "dpkg --install /tmp/libisl23_0.25-1.1_amd64.deb",
                "libisl23",
            ),
            (
                "dpkg -i /var/cache/apt/archives/*.deb",
                "local .deb batch",
            ),
        ],
    )
    def test_blocks_manual_deb_dependency_chasing_before_execution(self, command, package):
        tool = ShellTool()
        result = tool.execute(command)

        assert result.success is False
        assert result.metadata["blocked_by"] == "manual_deb_dependency_chase_guard"
        assert package in result.error
        assert "dpkg dependency loops" in result.error

    @pytest.mark.parametrize(
        "command",
        [
            "dpkg -I /tmp/package.deb",
            "dpkg -l | head -20",
            "dpkg -i /tmp/task-helper_1.0_amd64.deb",
        ],
    )
    def test_manual_deb_dependency_chase_guard_allows_probes_and_unrelated_debs(
        self, command
    ):
        assert manual_deb_dependency_chase_reason(command) is None

    def test_large_toolchain_install_plan_is_semantic_failure(self):
        output = (
            "The following NEW packages will be installed:\n"
            "  clang clang-14 libclang-cpp14 libllvm14 llvm-14 llvm-14-dev\n"
            "101 newly installed, 0 to remove and 38 not upgraded.\n"
            "Need to get 161 MB of archives.\n"
            "After this operation, 884 MB of additional disk space will be used.\n"
        )
        assert large_toolchain_install_plan_reason(output)

        result = ShellTool().execute(f"printf '%s' {output!r}")

        assert result.success is False
        assert result.metadata["semantic_failure_detected"] is True
        assert result.metadata["semantic_failure_kind"] == "large_toolchain_install_plan"
        assert "large compiler/toolchain install plan" in result.error

    def test_large_graphics_runtime_install_plan_is_semantic_failure(self):
        output = (
            "The following NEW packages will be installed:\n"
            "  libdrm-amdgpu1 libdrm-common libdrm-intel1 libdrm2 libgbm1 libgl1\n"
            "  libgl1-mesa-dri libglvnd0 libglx-mesa0 libglx0 libllvm19 libpciaccess0\n"
            "  libsensors-config libsensors5 libvulkan1 libwayland-client0\n"
            "  libwayland-server0 libx11-xcb1 libxcb-dri3-0 libxcb-glx0\n"
            "  libxcb-present0 libxcb-randr0 libxcb-sync1 libxcb-xfixes0\n"
            "  libxshmfence1 libxxf86vm1 libz3-4 mesa-libgallium mesa-vulkan-drivers\n"
            "0 upgraded, 29 newly installed, 0 to remove and 118 not upgraded.\n"
            "Need to get 60.1 MB of archives.\n"
            "After this operation, 289 MB of additional disk space will be used.\n"
        )
        assert large_graphics_runtime_install_plan_reason(output)
        assert large_toolchain_install_plan_reason(output)

        result = ShellTool().execute(f"printf '%s' {output!r}")

        assert result.success is False
        assert result.metadata["semantic_failure_detected"] is True
        assert result.metadata["semantic_failure_kind"] == "large_graphics_runtime_install_plan"
        assert "large graphics/CV runtime install plan" in result.error
        assert "Mesa/OpenGL/Vulkan/OpenCV" in result.error

    def test_cross_toolchain_install_plan_uses_lower_size_threshold(self):
        output = (
            "The following NEW packages will be installed:\n"
            "  binutils-mipsel-linux-gnu gcc-mipsel-linux-gnu linux-libc-dev-mipsel-cross\n"
            "21 newly installed, 0 to remove and 16 not upgraded.\n"
            "Need to get 28.7 MB/36.1 MB of archives.\n"
            "After this operation, 124 MB of additional disk space will be used.\n"
        )

        assert large_toolchain_install_plan_reason(output)

    def test_large_package_install_plan_is_semantic_failure(self):
        output = (
            "The following NEW packages will be installed:\n"
            "  r-cran-rstan r-cran-stanheaders r-cran-rcppparallel r-cran-rcppeigen\n"
            "327 newly installed, 0 to remove and 12 not upgraded.\n"
            "Need to get 412 MB of archives.\n"
            "After this operation, 2191 MB of additional disk space will be used.\n"
        )
        assert large_package_install_plan_reason(output)

        result = ShellTool().execute(f"printf '%s' {output!r}")

        assert result.success is False
        assert result.metadata["semantic_failure_detected"] is True
        assert result.metadata["semantic_failure_kind"] == "large_package_install_plan"
        assert "large transitive package install plan" in result.error

    def test_small_package_install_plan_is_not_semantic_failure(self):
        output = (
            "The following NEW packages will be installed:\n"
            "  jq libjq1 libonig5\n"
            "3 newly installed, 0 to remove and 0 not upgraded.\n"
            "Need to get 850 kB of archives.\n"
            "After this operation, 3 MB of additional disk space will be used.\n"
        )
        assert large_package_install_plan_reason(output) is None

        result = ShellTool().execute(f"printf '%s' {output!r}")

        assert result.success is True
        assert "semantic_failure_detected" not in result.metadata

    def test_timeout(self):
        tool = ShellTool(timeout_seconds=0.1)
        result = tool.execute("sleep 5", timeout=0.1)
        assert result.success is False
        assert "timed out" in result.error.lower()
        assert "operation timeout" in result.error
        assert "loop stop condition" in result.error
        assert "master, sub-agent, or Worker loop stop condition" in result.error
        assert "agent time budget" not in result.error
        assert result.metadata["operation_timeout_stop_condition"] is False
        assert result.metadata["timeout_seconds_stop_condition"] is False
        assert result.metadata["loop_stop_condition"] is False
        assert result.metadata["tool_timeout_telemetry"] is True
        assert result.metadata["tool_timeout_telemetry_source"] == "shell"
        assert result.metadata["tool_timeout_telemetry_stop_condition"] is False
        assert result.metadata["timeout_telemetry_stop_condition"] is False
        assert result.metadata["elapsed_ms"] > 0
        assert result.metadata["stdout_len"] == 0
        assert result.metadata["stderr_len"] == 0

    def test_verify_timeout_is_operation_metadata_not_loop_stop(self):
        tool = VerifyTool(timeout_seconds=0.1)
        result = tool.execute("sleep 5", timeout=0.1)

        assert result.success is False
        assert "verification timed out" in result.error
        assert "operation timeout" in result.error
        assert "loop stop condition" in result.error
        assert result.metadata["timed_out"] is True
        assert result.metadata["operation_timeout_stop_condition"] is False
        assert result.metadata["timeout_seconds_stop_condition"] is False
        assert result.metadata["loop_stop_condition"] is False
        assert result.metadata["tool_timeout_telemetry"] is True
        assert result.metadata["tool_timeout_telemetry_source"] == "verify"
        assert result.metadata["tool_timeout_telemetry_stop_condition"] is False
        assert result.metadata["timeout_telemetry_stop_condition"] is False
        assert result.metadata["elapsed_ms"] > 0

    def test_verify_detects_threshold_failure_hidden_by_success_exit(self):
        tool = VerifyTool()
        output = (
            "Verification of my_warrior.red:\n"
            "=== All opponent tests:\n"
            "  stone: 69 wins (Results: 69 29 2)\n"
            "  paper: 93 wins (Results: 93 0 7)\n"
            "  vampire: 90 wins (Results: 90 7 3)\n"
            "  snake: 50 wins (Results: 50 40 10)\n"
            "  g2-clear: 68 wins (Results: 68 30 2)\n"
        )
        result = tool.execute(f"printf '%b' {output!r}", timeout=5)

        assert result.success is False
        assert result.metadata["exit_code"] == 0
        assert result.metadata["semantic_failure_detected"] is True
        assert result.metadata["semantic_failure_kind"] == "corewar_threshold_failure"
        assert result.metadata["verification_semantic_failure_stop_condition"] is False
        assert result.metadata["loop_stop_condition"] is False
        assert "verification output indicates an unmet threshold" in result.error
        assert "stone: 69 wins" in result.output

    def test_verify_detects_explicit_only_achieved_threshold_failure(self):
        tool = VerifyTool()
        output = "AssertionError: Only achieved 52% win rate vs stone.red (need 75%+)"
        result = tool.execute(f"printf '%b' {output!r}", timeout=5)

        assert result.success is False
        assert result.metadata["semantic_failure_kind"] == "verification_threshold_failure"
        assert "unmet threshold" in result.error

    def test_verify_detects_regex_backreference_failure_hidden_by_success_exit(self):
        tool = VerifyTool()
        output = (
            "fen = re.sub(pattern, repl, fen)\n"
            "E           re.PatternError: invalid group reference 10 at position 19\n"
            "Error at pair 6174: invalid group reference 2 at position 5\n"
            "  Pattern: '(?m) (w|b) [^-]+ [a-h][36] '\n"
            "  Replacement: ' \\1 \\2 - '\n"
        )
        result = tool.execute(f"printf '%s' {shlex.quote(output)}", timeout=5)

        assert result.success is False
        assert result.metadata["exit_code"] == 0
        assert result.metadata["semantic_failure_detected"] is True
        assert (
            result.metadata["semantic_failure_kind"]
            == "regex_replacement_backreference_failure"
        )
        assert "invalid Python re.sub replacement backreference" in result.error

    def test_verify_regex_backreference_helper_ignores_plain_log_inspection(self):
        assert verify_semantic_failure_kind(
            "E re.PatternError: invalid group reference 10 at position 19",
            command="cat prior-verifier.log",
            returncode=0,
        ) is None

    def test_verify_detects_heavy_ml_cv_import_failure_hidden_by_success_exit(self):
        tool = VerifyTool()
        output = (
            "/usr/local/lib/python3.11/site-packages/cv2/__init__.py: line 181\n"
            "ImportError: libGL.so.1: cannot open shared object file: No such file or directory\n"
        )
        result = tool.execute(f"printf '%s' {shlex.quote(output)}", timeout=5)

        assert result.success is False
        assert result.metadata["exit_code"] == 0
        assert result.metadata["semantic_failure_detected"] is True
        assert result.metadata["semantic_failure_kind"] == "heavy_ml_cv_import_failure"
        assert "heavy ML/CV import output" in result.error
        assert "CSV/image contract" in result.error

    def test_verify_detects_numpy_eigensolver_failure_hidden_by_success_exit(self):
        tool = VerifyTool()
        output = (
            "from eigen import find_dominant_eigenvalue_and_eigenvector\n"
            "numpy._core._exceptions._UFuncOutputCastingError: Cannot cast "
            "ufunc 'subtract' output from dtype('complex128') to dtype('float64') "
            "with casting rule 'same_kind'\n"
        )
        result = tool.execute(f"printf '%s' {shlex.quote(output)}", timeout=5)

        assert result.success is False
        assert result.metadata["exit_code"] == 0
        assert result.metadata["semantic_failure_detected"] is True
        assert result.metadata["semantic_failure_kind"] == "numpy_eigensolver_failure"
        assert "NumPy eigensolver output" in result.error
        assert "eval.py" in result.error

    def test_verify_detects_numpy_eigensolver_speed_failure_hidden_by_success_exit(self):
        tool = VerifyTool()
        output = (
            "test_outputs.py::test_speedup\n"
            "size = 8\n"
            "from eigen import find_dominant_eigenvalue_and_eigenvector\n"
            ">       assert dt < ref_dt, f\"{dt:.6f} seconds/call > {ref_dt:.6f} seconds/call\"\n"
            "E       AssertionError: 0.000026 seconds/call > 0.000025 seconds/call\n"
        )
        result = tool.execute(f"printf '%s' {shlex.quote(output)}", timeout=5)

        assert result.success is False
        assert result.metadata["exit_code"] == 0
        assert result.metadata["semantic_failure_detected"] is True
        assert (
            result.metadata["semantic_failure_kind"]
            == "numpy_eigensolver_speed_threshold_failure"
        )
        assert "slower than the reference" in result.error
        assert "timing harness" in result.error

    def test_verify_detects_single_file_deliverable_directory_failure_hidden_by_success_exit(self):
        tool = VerifyTool()
        output = (
            "def test_fibonacci_polyglot():\n"
            "    # Verify that only main.rs exists in /app/polyglot\n"
            "    polyglot_files = os.listdir(\"/app/polyglot\")\n"
            ">       assert polyglot_files == [\"main.rs\"], (\n"
            "E       AssertionError: Expected only main.rs, found: ['test_poly.rs', 'main.rs', 'cmain']\n"
        )
        result = tool.execute(f"printf '%s' {shlex.quote(output)}", timeout=5)

        assert result.success is False
        assert result.metadata["exit_code"] == 0
        assert result.metadata["semantic_failure_detected"] is True
        assert (
            result.metadata["semantic_failure_kind"]
            == "single_file_deliverable_directory_contract"
        )
        assert "single-file deliverable directory" in result.error
        assert "compiler/interpreter" in result.error

    def test_verify_detects_gpt2_codegolf_wrong_output_hidden_by_success_exit(self):
        tool = VerifyTool()
        output = (
            "def test_gpt2_implementation():\n"
            "    gpt2_path = Path(\"/app/gpt2.c\")\n"
            "    compile_result = subprocess.run([\"gcc\", \"-O3\", \"/app/gpt2.c\", \"-lm\"])\n"
            "    run_result = subprocess.run([\"/app/a.out\", \"gpt2-124M.ckpt\", "
            "\"vocab.bpe\", 'THIS SOFTWARE IS PROVIDED \"AS IS\", WITHOUT'], "
            "capture_output=True, text=True, timeout=90)\n"
            ">       assert \"WARRANTY OF ANY KIND, EXPRESS OR IMPLIED\" in run_result.stdout\n"
            "E       AssertionError: Wrong output\n"
            "E       assert 'WARRANTY OF ANY KIND, EXPRESS OR IMPLIED' in '[46107][46107][46107]\\n'\n"
        )
        result = tool.execute(f"printf '%s' {shlex.quote(output)}", timeout=5)

        assert result.success is False
        assert result.metadata["exit_code"] == 0
        assert result.metadata["semantic_failure_detected"] is True
        assert result.metadata["semantic_failure_kind"] == "gpt2_codegolf_text_contract"
        assert "GPT2 codegolf text contract" in result.error
        assert "repeated token ids" in result.error
        assert "90s timeout" in result.error

    def test_verify_detects_structured_csv_table_key_failure_hidden_by_success_exit(self):
        tool = VerifyTool()
        output = (
            "def test_cell_metadata_csv():\n"
            "    df = pd.read_csv(args.csv_path)\n"
            "    for _, row in df.iterrows():\n"
            ">       assert row['cell_id'] in expected_data\n"
            "E       AssertionError: unexpected row key\n"
            "E       assert 'cell-17' in expected_data\n"
        )
        result = tool.execute(f"printf '%s' {shlex.quote(output)}", timeout=5)

        assert result.success is False
        assert result.metadata["exit_code"] == 0
        assert result.metadata["semantic_failure_detected"] is True
        assert result.metadata["semantic_failure_kind"] == "structured_csv_table_contract"
        assert "structured CSV/table contract" in result.error
        assert "key or identifier values" in result.error
        assert "expected keyed row content" in result.error

    def test_verify_detects_missing_output_artifact_hidden_by_success_exit(self):
        tool = VerifyTool()
        output = (
            "def test_recovery_file_exists():\n"
            "    recovered_pass_path = Path(\"/app/recovered_passwords.txt\")\n"
            ">       assert recovered_pass_path.exists(), \"Recovery file not found\"\n"
            "E       AssertionError: Recovery file not found\n"
            "E        + where exists = PosixPath('/app/recovered_passwords.txt').exists\n"
        )
        result = tool.execute(f"printf '%s' {shlex.quote(output)}", timeout=5)

        assert result.success is False
        assert result.metadata["exit_code"] == 0
        assert result.metadata["semantic_failure_detected"] is True
        assert result.metadata["semantic_failure_kind"] == "missing_output_artifact_contract"
        assert "missing output artifact contract" in result.error
        assert "test -s" in result.error
        assert "artifact-wide searches" in result.error

    def test_verify_detects_dna_assembly_primer_failure_hidden_by_success_exit(self):
        tool = VerifyTool()
        output = (
            "def test_primers():\n"
            "    primers_path = Path(\"/app/primers.fasta\")\n"
            "    assert len(lines) == 16, \"Invalid number of lines in primers.fasta.\"\n"
            "    assert all(k in primers for k in [\"input_fwd\", \"input_rev\", "
            "\"egfp_fwd\", \"egfp_rev\", \"flag_fwd\", \"flag_rev\", "
            "\"snap_fwd\", \"snap_rev\"])\n"
            "    def parse_bsai_primer(primer):\n"
            "        \"\"\"Primer (5'->3'): [clamp] ggtctc [oooo] [binding]\"\"\"\n"
            "        site = \"ggtctc\"\n"
            "        i = primer.find(site)\n"
            ">       assert i >= 1, \"Primer must have clamp of at least 1 nucleotide before BsaI site.\"\n"
            "E       AssertionError: Primer must have clamp of at least 1 nucleotide before BsaI site.\n"
        )
        result = tool.execute(f"printf '%s' {shlex.quote(output)}", timeout=5)

        assert result.success is False
        assert result.metadata["exit_code"] == 0
        assert result.metadata["semantic_failure_detected"] is True
        assert result.metadata["semantic_failure_kind"] == "dna_assembly_primer_contract"
        assert "DNA assembly primer contract" in result.error
        assert "ggtctc/BsaI" in result.error
        assert "four-base overhang" in result.error
        assert "parse_bsai_primer/make_fragment" in result.error

    def test_verify_success_threshold_output_remains_success(self):
        tool = VerifyTool()
        output = (
            "vs stone.red   :  78/100 wins ( 78%) - PASS (need 75%+)\n"
            "vs vampire.red :  79/100 wins ( 79%) - PASS (need 75%+)\n"
            "PASSED (5/5): stone.red, vampire.red, paper.red, snake.red, g2-clear.red\n"
            "FAILED (0/5): None\n"
        )
        result = tool.execute(f"printf '%b' {output!r}", timeout=5)

        assert result.success is True
        assert result.metadata == {"exit_code": 0}

    def test_verify_blocks_nested_external_agent_creation_before_exec(self):
        tool = VerifyTool()
        result = tool.execute("c='codex exec'; $c fix", timeout=5)

        assert result.success is False
        assert result.metadata["blocked_by"] == "nested_sub_agent_creation_guard"
        assert result.metadata["sub_agent_creation_guard"] is True
        assert result.metadata["nested_sub_agent_creation_allowed"] is False
        assert result.metadata["only_master_loop_may_create_sub_agents"] is True
        _assert_policy_guard_is_non_terminal(result.metadata)
        assert "only the master HL orchestrator may create sub-agents" in result.error

    def test_verify_semantic_failure_helper_ignores_plain_failure_log_inspection(self):
        assert verify_semantic_failure_kind(
            "Earlier verifier said FAILED (0/5): None in a copied note",
            command="cat notes.txt",
            returncode=0,
        ) is None

    def test_shell_timeout_records_command_telemetry_even_without_partial_output(self):
        tool = ShellTool(timeout_seconds=0.2)
        result = tool.execute("sleep 5", timeout=0.2)

        assert result.success is False
        assert result.metadata["tool_timeout_telemetry"] is True
        assert result.metadata["partial_output_available"] is False
        assert result.metadata["stdout_len"] == 0
        assert result.metadata["stderr_len"] == 0
        assert result.metadata["stdout_tail"] == ""
        assert result.metadata["stderr_tail"] == ""
        assert result.metadata["tool_timeout_telemetry_stop_condition"] is False

    @pytest.mark.parametrize(
        "command",
        [
            "apt-get install -y gcc",
            "apt-cache search mipsel",
            "pip install fasttext 2>&1 | tail -20",
            "R -e 'install.packages(\"rstan\", repos=\"https://cloud.r-project.org\")'",
        ],
    )
    def test_package_manager_timeout_is_capped(self, command):
        effective, note = package_manager_timeout_cap(command, 120)

        assert effective == 60
        assert "Package-manager command timeout was capped" in note
        assert "operation-level evidence" in note
        assert "loop stop condition" in note
        assert "master, sub-agent, or Worker loop stop condition" in note
        assert "agent time budget" not in note

    def test_package_manager_timeout_cap_is_reported_as_operation_metadata(self):
        tool = ShellTool(timeout_seconds=120)
        result = tool.execute("printf ok; # apt-cache search mipsel", timeout=120)

        assert result.metadata["timeout_seconds"] == 60
        assert result.metadata["requested_timeout_seconds"] == 120
        assert result.metadata["timeout_capped"] is True
        assert result.metadata["operation_timeout_stop_condition"] is False
        assert result.metadata["timeout_seconds_stop_condition"] is False
        assert result.metadata["loop_stop_condition"] is False
        assert "operation-level evidence" in result.output

    def test_non_package_timeout_is_not_capped(self):
        effective, note = package_manager_timeout_cap("python3 train.py", 120)

        assert effective == 120
        assert note == ""

    def test_blocks_terminal_bench_leaderboard_access(self):
        tool = ShellTool()
        result = tool.execute("curl https://www.tbench.ai/leaderboard/terminal-bench/2.0")
        assert result.success is False
        assert result.metadata["blocked_by"] == "leaderboard_integrity_guard"
        assert "Terminal-Bench website access" in result.error

    def test_blocks_host_hl_memory_search_before_execution(self):
        tool = ShellTool()
        result = tool.execute("find /trials/runs -name trajectory.jsonl | head -20")

        assert result.success is False
        assert result.metadata["blocked_by"] == "host_memory_guard"
        assert result.metadata["blocked_reason"] == "host_memory_search"
        _assert_policy_guard_is_non_terminal(result.metadata)
        assert "Same-task memory summaries" in result.error

    def test_allows_task_local_trajectory_filename(self):
        tool = ShellTool()
        result = tool.execute("printf 'trajectory.jsonl\n'")

        assert result.success is True
        assert "trajectory.jsonl" in result.output

    def test_blocks_background_package_manager_commands(self):
        tool = ShellTool()
        result = tool.execute(
            "pip install httpstan > /tmp/pip.log 2>&1 & echo started"
        )

        assert result.success is False
        assert result.metadata["blocked_by"] == "background_package_command_guard"
        _assert_policy_guard_is_non_terminal(result.metadata)
        assert "can outlive the Worker tool timeout" in result.error

    def test_package_stderr_redirection_is_not_background_operator(self):
        assert background_package_command_reason("pip install numpy 2>&1 | tail -5") is None

    @pytest.mark.parametrize(
        "command",
        [
            "dpkg --configure -a >/tmp/dpkg.log 2>&1 &",
            "R CMD INSTALL /tmp/rstan.tar.gz > /tmp/r-install.log 2>&1 &",
        ],
    )
    def test_blocks_background_package_setup_beyond_pip_and_apt(self, command):
        tool = ShellTool()
        result = tool.execute(command)

        assert result.success is False
        assert result.metadata["blocked_by"] == "background_package_command_guard"
        assert "can outlive the Worker tool timeout" in result.error

    @pytest.mark.parametrize(
        "command",
        [
            "rm -f /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock",
            "fuser -k /var/lib/apt/lists/lock",
            "kill $(pgrep -f apt-get)",
        ],
    )
    def test_blocks_package_lock_and_process_cleanup(self, command):
        tool = ShellTool()
        result = tool.execute(command)

        assert result.success is False
        assert result.metadata["blocked_by"] == "background_package_command_guard"
        assert "lock or process cleanup" in result.error

    def test_blocks_r_package_lock_cleanup(self):
        tool = ShellTool()
        result = tool.execute(
            "rm -rf /usr/local/lib/R/site-library/00LOCK-rstan && "
            "R CMD INSTALL /tmp/rstan_2.32.7.tar.gz"
        )

        assert result.success is False
        assert result.metadata["blocked_by"] == "background_package_command_guard"
        assert "00LOCK cleanup" in result.error
        _assert_policy_guard_is_non_terminal(result.metadata)

    @pytest.mark.parametrize(
        "command",
        [
            "dpkg --configure -a 2>&1 | head -20",
            "DEBIAN_FRONTEND=noninteractive timeout 120 apt-get install -f -y 2>&1",
            "apt --fix-broken install -y",
        ],
    )
    def test_blocks_package_manager_state_repair_loops(self, command):
        tool = ShellTool()
        result = tool.execute(command)

        assert result.success is False
        assert result.metadata["blocked_by"] == "background_package_command_guard"
        assert "broken-state repair" in result.error

    @pytest.mark.parametrize(
        "command",
        [
            (
                "python3 -c \"import urllib.request; "
                "urllib.request.urlopen('https://files.pythonhosted.org/pkg.whl')\""
            ),
            (
                "curl -s --connect-timeout 10 -k "
                "'http://cran.r-project.org/src/contrib/rstan_2.32.7.tar.gz' "
                "-o /tmp/rstan.tar.gz"
            ),
            (
                "python3 - <<'PY'\nimport urllib.request\n"
                "urllib.request.urlretrieve('http://deb.debian.org/debian/pool/main/libx/libx.deb', '/tmp/libx.deb')\nPY"
            ),
            (
                "timeout 10 curl -s -k "
                "'http://archive.ubuntu.com/ubuntu/pool/universe/r/r-cran-stanheaders/"
                "r-cran-stanheaders_2.32.5-1_amd64.deb' -o /tmp/r-cran-stanheaders.deb"
            ),
            (
                "python3 -c \"import urllib.request; "
                "urllib.request.urlopen('https://mirrors.tuna.tsinghua.edu.cn/pypi/web/"
                "packages/d0/httpstan-4.13.0-cp311-cp311-linux_x86_64.whl')\""
            ),
            (
                "curl -L 'https://mirrors.aliyun.com/debian/pool/main/g/gcc-12/"
                "gcc-12_12.2.0-14_amd64.deb' -o /tmp/gcc.deb"
            ),
            (
                "python3 -c \"import urllib.request; "
                "urllib.request.urlopen('http://ftp.debian.org/debian/pool/main/g/gcc-12/"
                "gcc-12_12.2.0-14+deb12u1_amd64.deb')\""
            ),
            "R -e 'available.packages(repos=\"http://cran.r-project.org\")'",
        ],
    )
    def test_blocks_manual_dependency_downloads_before_execution(self, command):
        tool = ShellTool()
        result = tool.execute(command)

        assert result.success is False
        assert result.metadata["blocked_by"] == "manual_dependency_download_guard"
        assert "hand-written dependency downloads" in result.error

    @pytest.mark.parametrize(
        "command",
        [
            (
                "python3 <<'PY'\n"
                "import pip._internal.network.session\n"
                "import pip._internal.cli.main as pip_main\n"
                "pip_main.main()  # --break-system-packages files.pythonhosted.org httpstan\n"
                "PY"
            ),
            (
                "python3 <<'PY'\n"
                "import pip._internal.cli.main as pip_main\n"
                "pip_main.main(['install', '-i', 'https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple', 'httpstan'])\n"
                "PY"
            ),
        ],
    )
    def test_blocks_direct_scripted_package_manager_command_before_execution(self, command):
        tool = ShellTool()
        result = tool.execute(command)

        assert result.success is False
        assert result.metadata["blocked_by"] == "scripted_package_manager_guard"
        assert "inline script wraps package-manager" in result.error
        assert scripted_package_manager_command_reason(command)

    def test_scripted_package_manager_guard_allows_safe_task_data_script(self):
        command = (
            "python3 <<'PY'\n"
            "import urllib.request\n"
            "urllib.request.urlretrieve('https://example.com/task-data.tar.gz', '/tmp/data.tar.gz')\n"
            "PY"
        )

        assert scripted_package_manager_command_reason(command) is None

    @pytest.mark.parametrize(
        "command",
        [
            (
                "timeout 5 python3 -c \"import urllib.request; "
                "r=urllib.request.urlopen('http://google.com', timeout=5); print(r.status)\""
            ),
            (
                "python3 -c \"import urllib.request; "
                "urllib.request.urlretrieve('https://github.com/yt-dlp/yt-dlp/"
                "releases/latest/download/yt-dlp','/usr/local/bin/yt-dlp')\""
            ),
            "curl -L https://example.com/task-data.tar.gz -o /tmp/task-data.tar.gz",
        ],
    )
    def test_manual_dependency_download_guard_allows_non_package_fetches(self, command):
        assert manual_dependency_download_reason(command) is None

    def test_allows_foreground_bounded_package_manager_text(self):
        tool = ShellTool()
        result = tool.execute("printf 'pip install should be run in foreground\\n'")

        assert result.success is True

    def test_blocks_unbounded_root_find_commands(self):
        tool = ShellTool()
        result = tool.execute('find / -name "*.whl" -type f 2>/dev/null | head -20')

        assert result.success is False
        assert result.metadata["blocked_by"] == "broad_root_find_guard"
        assert "system-prefix searches" in result.error
        assert "Narrow the search" in result.error
        assert "single-operation evidence window" in result.error
        assert "loop stop condition" in result.error
        assert "master, sub-agent, or Worker loop stop condition" in result.error
        assert "tool budget" not in result.error.lower()

    @pytest.mark.parametrize(
        "command",
        [
            'find /usr -name "*mips*" -type f 2>/dev/null | head -20',
            'find /opt -type f -executable 2>/dev/null | head -20',
            'find /root -name "*.whl" 2>/dev/null | head -20',
        ],
    )
    def test_blocks_unbounded_system_prefix_find_commands(self, command):
        tool = ShellTool()
        result = tool.execute(command)

        assert result.success is False
        assert result.metadata["blocked_by"] == "broad_root_find_guard"
        assert "system-prefix searches" in result.error

    def test_blocks_unbounded_root_find_after_shell_separator(self):
        command = (
            "apt-cache search mips 2>/dev/null | head -20; "
            "dpkg -l 2>/dev/null | grep -i mips; "
            "find / -name 'mipsel-linux-gnu-gcc*' -o "
            "-name 'mips-linux-gnu-gcc*' 2>/dev/null | head -10"
        )

        assert "system-prefix searches" in broad_root_find_command_reason(command)
        assert "tool budget" not in broad_root_find_command_reason(command).lower()

    @pytest.mark.parametrize(
        "command",
        [
            "cat /proc/*/cmdline 2>/dev/null | tr '\\0' ' ' | head -50",
            "ls /proc/[0-9]*/fd/ 2>/dev/null | head -20",
            "for pid in $(ls /proc/ | grep -E '^[0-9]+$'); do cat /proc/$pid/cmdline; done",
        ],
    )
    def test_blocks_broad_proc_process_scans(self, command):
        reason = broad_proc_scan_command_reason(command)
        assert reason is not None
        assert "proc" in reason
        assert "single-operation evidence window" in reason
        assert "loop stop condition" in reason
        assert "master, sub-agent, or Worker loop stop condition" in reason
        assert "tool budget" not in reason.lower()

        tool = ShellTool()
        result = tool.execute(command)

        assert result.success is False
        assert result.metadata["blocked_by"] == "broad_proc_scan_guard"
        assert "specific known PID" in result.error
        assert "tool budget" not in result.error.lower()

    @pytest.mark.parametrize(
        "command",
        [
            "cat /proc/1/cmdline | tr '\\0' ' '",
            "cat /proc/net/tcp | head -5",
            "ps -eo pid,cmd | head -20",
            "pgrep -af 'python3.*http.server'",
        ],
    )
    def test_allows_targeted_proc_and_process_checks(self, command):
        assert broad_proc_scan_command_reason(command) is None

    @pytest.mark.parametrize(
        "command",
        [
            'find / -maxdepth 2 -name "*.whl"',
            'find /app -name "*.py"',
            'find /tmp/hl-verifier-cache -name "numpy*" | head -20',
        ],
    )
    def test_allows_bounded_find_commands(self, command):
        assert broad_root_find_command_reason(command) is None

    def test_blocks_external_solution_url_to_agents_md(self):
        tool = ShellTool()
        result = tool.execute("curl https://example.com/solutions/task-a.py > AGENTS.md")
        assert result.success is False
        assert result.metadata["blocked_by"] == "leaderboard_integrity_guard"
        assert "External solution URL access" in result.error

    @pytest.mark.parametrize(
        "command",
        [
            "codex --help",
            "codex -V",
            "codex exec --json 'fix this'",
            "openai-codex exec --json 'fix this'",
            "OPENAI_API_KEY=x codex run 'fix this'",
            "aider --message 'fix this'",
            "amp run 'fix this'",
            "claude --print 'fix this'",
            "claude-code --help",
            "cursor-agent 'fix this'",
            "forgecode run task",
            "factory mission run",
            "droid mission run",
            "factory-droid mission run",
            "gemini --prompt 'fix this'",
            "gemini-cli --prompt 'fix this'",
            "opencode run 'fix this'",
            "timeout 30 codex exec --json 'fix this'",
            "command codex --help",
            "sudo -E codex exec --json 'fix this'",
            "sudo --user nobody codex exec --json 'fix this'",
            "env -i codex exec --json 'fix this'",
            "env -u OPENAI_API_KEY codex exec --json 'fix this'",
            "env --chdir=/tmp codex exec --json 'fix this'",
            "env --ignore-environment codex exec --json 'fix this'",
            "exec codex exec --json 'fix this'",
            "nohup codex exec --json 'fix this'",
            "setsid codex exec --json 'fix this'",
            "nice -n 10 codex exec --json 'fix this'",
            "time codex exec --json 'fix this'",
            "/usr/bin/time -f %E codex exec --json 'fix this'",
            "stdbuf -oL codex exec --json 'fix this'",
            "bash -lc 'codex exec --json fix'",
            "bash -c codex --help",
            "bash -lc codex --help",
            "bash -o pipefail -c codex --help",
            "sh -c 'claude --print fix'",
            "sh -c claude --print",
            "env FOO=bar bash -lc 'forgecode run task'",
            "npx --yes codex exec 'fix this'",
            "pnpm dlx codex exec 'fix this'",
            "uvx --from openai codex exec 'fix this'",
            "uvx openai-codex exec 'fix this'",
            "pipx run codex exec 'fix this'",
            "pipx run --spec openai-codex codex exec 'fix this'",
            "pipx run openai-codex exec 'fix this'",
            "uv tool run codex exec 'fix this'",
            "poetry run codex exec 'fix this'",
            "pipenv run codex exec 'fix this'",
            "conda run -n base codex exec 'fix this'",
            "bun x codex exec 'fix this'",
            "python -m codex --help",
            "python -m codex.cli exec 'fix this'",
            "python -m openai.codex exec 'fix this'",
            "python -m openai_codex exec 'fix this'",
            "python -m opencode run 'fix this'",
            "python -m aider --message 'fix this'",
            "python -m gemini --prompt 'fix this'",
            "python -c \"import subprocess; subprocess.run(['codex', 'exec', 'fix'])\"",
            "python -c \"import subprocess; subprocess.run(['openai-codex', 'exec', 'fix'])\"",
            "python -c \"import subprocess; subprocess.run(['claude-code', '--print', 'fix'])\"",
            "python -c \"import runpy; runpy.run_module('codex.cli', run_name='__main__')\"",
            "python -c \"from runpy import run_module; run_module('openai.codex', run_name='__main__')\"",
            "python -c \"import importlib; importlib.import_module('codex.cli').main()\"",
            "python -c \"__import__('openai.codex').codex.main()\"",
            "python -c \"import subprocess as sp; sp.run(['codex', 'exec', 'fix'])\"",
            "python -c \"from subprocess import run as rr; rr(['codex', 'exec', 'fix'])\"",
            "python -c \"import subprocess; getattr(subprocess, 'run')(['codex','exec','fix'])\"",
            "python -c \"import subprocess; getattr(subprocess, 'Popen')(['claude-code','--print','fix'])\"",
            "python -c \"__import__('subprocess').run(['opencode','run','fix'])\"",
            "python -c \"getattr(__import__('subprocess'), 'run')(['codex','exec','fix'])\"",
            "python -c \"import subprocess; rr = subprocess.run; rr(['codex','exec','fix'])\"",
            "python -c \"import importlib; importlib.import_module('subprocess').run(['codex','exec','fix'])\"",
            "python -c \"from importlib import import_module; import_module('subprocess').run(['opencode','run','fix'])\"",
            "python -c \"import os as o; o.system('opencode run fix')\"",
            "python -c \"import os; getattr(os, 'system')('codex exec fix')\"",
            "node -e \"require('child_process').spawn('codex',['exec','fix'])\"",
            "node -e \"require('child_process').spawn('openai-codex',['exec','fix'])\"",
            "node -e \"require('child_process').spawn('factory',['mission','run'])\"",
            "node -e \"require('child_process').spawn('droid',['mission','run'])\"",
            "node -e \"require('node:child_process').execFile('opencode',['run','fix'])\"",
            "node -e \"require('child_process').spawnSync('codex',['exec','fix'])\"",
            "node -e \"const cp = require('child_process'); cp.spawn('codex',['exec','fix'])\"",
            "node -e \"const cp = require('child_process'); cp.spawn('factory',['mission','run'])\"",
            "node -e \"const cp = require('child_process'); cp.execSync('codex exec fix')\"",
            "node -e \"const {spawn: s} = require('child_process'); s('opencode',['run','fix'])\"",
            "node -e \"const {spawn: s} = require('child_process'); s('droid',['mission','run'])\"",
            "node -e \"const {execFileSync: efs} = require('node:child_process'); efs('opencode',['run','fix'])\"",
            "node -e \"import {spawnSync} from 'child_process'; spawnSync('codex',['exec','fix'])\"",
            "node -e \"import {spawnSync} from 'child_process'; spawnSync('factory',['mission','run'])\"",
            "node -e \"import {spawn as s} from 'node:child_process'; s('opencode',['run','fix'])\"",
            "node -e \"import * as cp from 'node:child_process'; cp.execSync('codex exec fix')\"",
            "ruby -e \"system 'opencode run fix'\"",
            "ruby -e \"spawn 'codex exec fix'\"",
            "ruby -e \"spawn 'openai-codex exec fix'\"",
            "ruby -e \"spawn 'factory mission run'\"",
            "ruby -e \"spawn 'droid mission run'\"",
            "ruby -e \"send(:system, 'codex exec fix')\"",
            "ruby -e \"send(:system, 'factory mission run')\"",
            "ruby -e \"Kernel.send(:spawn, 'opencode run fix')\"",
            "ruby -e \"Kernel.send(:spawn, 'droid mission run')\"",
            "printf fix | xargs codex exec",
            "xargs -n 1 codex exec",
            "find . -exec codex exec fix ;",
            "find . -execdir claude --print fix ;",
            "script -q -c 'codex exec fix' /dev/null",
            "script -q -c codex /dev/null",
            "watch -n 1 codex exec fix",
            "parallel --jobs 2 codex exec ::: fix",
            "eval codex exec fix",
            "eval 'bash -lc \\\"codex exec fix\\\"'",
            "busybox sh -c 'codex exec fix'",
            "busybox sh -c codex --help",
            "if codex exec fix; then echo ok; fi",
            "f(){ codex exec fix; }; f",
            "function f { opencode run fix; }; f",
            "alias c='codex exec'; c fix",
            "c=codex; $c exec fix",
            "export c=codex; $c exec fix",
            "readonly c=codex; $c exec fix",
            "declare c=codex; $c exec fix",
            "local c=codex; $c exec fix",
            "c=$(printf codex); $c exec fix",
            "c=$(printf %s codex); $c exec fix",
            "c=$(echo codex); $c exec fix",
            "c=`echo codex`; $c exec fix",
            "c='codex exec'; $c fix",
            "env c=codex bash -lc '$c exec fix'",
            "bash -lc \"f(){ codex exec fix; }; f\"",
            "bash -lc \"alias c='codex exec'; c fix\"",
            "bash -lc \"c=codex; $c exec fix\"",
            "bash -lc \"export c=codex; $c exec fix\"",
            "echo $(codex exec fix)",
            "source <(codex exec fix)",
            "echo $(bash -lc 'codex exec fix')",
            "python -c \"import subprocess; cmd = 'cod' + 'ex'; subprocess.run([cmd, 'exec', 'fix'])\"",
            "python -c \"import subprocess; subprocess.run(['co' 'dex', 'exec', 'fix'])\"",
            "python -c \"import subprocess; cmd = 'co' 'dex'; subprocess.run([cmd, 'exec', 'fix'])\"",
            "python -c \"import subprocess; cmd = ''.join(['co','dex']); subprocess.run([cmd, 'exec', 'fix'])\"",
            "python -c \"import subprocess; cmd = f'codex'; subprocess.run([cmd, 'exec', 'fix'])\"",
            "python -c \"import subprocess; cmd = chr(99)+chr(111)+chr(100)+chr(101)+chr(120); subprocess.run([cmd, 'exec', 'fix'])\"",
            "python -c \"import subprocess; cmd = bytes([99,111,100,101,120]).decode(); subprocess.run([cmd, 'exec', 'fix'])\"",
            "python -c \"import subprocess; cmd = ('codex').replace('x','x'); subprocess.run([cmd, 'exec', 'fix'])\"",
            "python -c \"import subprocess; cmd = ('CODEX').lower(); subprocess.run([cmd, 'exec', 'fix'])\"",
            "python -c \"import subprocess; cmd = 'factory'; subprocess.run([cmd, 'mission', 'run'])\"",
            "python -c \"import subprocess; cmd = 'droid'; subprocess.run([cmd, 'mission', 'run'])\"",
            "node -e \"const c = 'cod' + 'ex'; require('child_process').spawn(c, ['exec','fix'])\"",
            "node -e \"const c = 'factory'; require('child_process').spawn(c, ['mission','run'])\"",
            "node -e \"const c = ['co','dex'].join(''); require('child_process').spawn(c, ['exec','fix'])\"",
            "node -e \"const c = String.fromCharCode(99,111,100,101,120); require('child_process').spawn(c, ['exec','fix'])\"",
            "node -e \"const c = Buffer.from([99,111,100,101,120]).toString(); require('child_process').spawn(c, ['exec','fix'])\"",
            "ruby -e \"c='cod'+'ex'; spawn c, 'exec', 'fix'\"",
            "ruby -e \"c='droid'; spawn c, 'mission', 'run'\"",
            "ruby -e \"c=['co','dex'].join; spawn c, 'exec', 'fix'\"",
            "ruby -e \"c=%q{codex}; spawn c, 'exec', 'fix'\"",
            "lua -e \"os.execute('codex exec fix')\"",
            "lua -e \"io.popen('codex exec fix')\"",
            "php -r \"exec('codex exec fix');\"",
            "php -r \"shell_exec('codex exec fix');\"",
            "printf '#!/bin/sh\\ncodex exec fix\\n' > /tmp/run_agent.sh",
            "printf '#!/bin/sh\\nopenai-codex exec fix\\n' > /tmp/run_agent.sh",
            "printf '#!/bin/sh\\ncodex exec fix\\n' | tee /tmp/run_agent.sh",
            "cat <<'SH' > /tmp/run_agent.sh\n#!/bin/sh\ncodex exec fix\nSH",
            "python -c \"from pathlib import Path; Path('/tmp/run_agent.sh').write_text('codex exec fix')\"",
            "python -c \"from pathlib import Path; Path('/tmp/run_agent.sh').write_text('openai-codex exec fix')\"",
            "node -e \"require('fs').writeFileSync('/tmp/run_agent.sh', 'codex exec fix')\"",
            "ruby -e \"File.write('/tmp/run_agent.sh', 'codex exec fix')\"",
        ],
    )
    def test_blocks_nested_external_agent_creation(self, command):
        assert external_agent_command_reason(command)

        result = ShellTool().execute(command)

        assert result.success is False
        assert result.metadata["blocked_by"] == "nested_sub_agent_creation_guard"
        assert result.metadata["sub_agent_creation_guard"] is True
        assert result.metadata["nested_sub_agent_creation_allowed"] is False
        assert result.metadata["only_master_loop_may_create_sub_agents"] is True
        assert result.metadata["sub_agent_creation_loop_stop_condition"] is False
        assert result.metadata["nested_sub_agent_creation_stop_condition"] is False
        _assert_policy_guard_is_non_terminal(result.metadata)
        assert "only the master HL orchestrator may create sub-agents" in result.error

    def test_allows_non_agent_codex_mentions(self):
        for command in (
            "printf 'codex exec is blocked by policy'",
            "printf 'openai-codex exec is blocked by policy'",
            "printf 'factory mission run is blocked by policy'",
            "echo 'droid mission run is blocked by policy'",
            "node -e \"require('child_process').spawn('echo',['factory mission run is blocked by policy'])\"",
            "ruby -e \"spawn 'echo droid mission run is blocked by policy'\"",
            "python -c \"import subprocess; cmd='echo'; subprocess.run([cmd, 'factory mission run is blocked'])\"",
        ):
            assert external_agent_command_reason(command) is None
        for command in (
            "printf 'codex exec is blocked by policy'",
            "printf 'openai-codex exec is blocked by policy'",
            "printf 'factory mission run is blocked by policy'",
            "echo 'droid mission run is blocked by policy'",
        ):
            result = ShellTool().execute(command)

            assert result.success is True

    def test_allows_non_agent_concatenated_text_mentions(self):
        command = "printf 'cod' + 'ex'"

        assert external_agent_command_reason(command) is None
        result = ShellTool().execute(command)

        assert result.success is True

    def test_allows_search_pattern_alternation_mentioning_agent_names(self):
        # A read-only search whose quoted regex pattern contains "|" must not be
        # split into shell segments; alternation tokens like "Codex update" are
        # search text, not a pipeline into an external agent CLI.
        for command in (
            'rg -n "HarnessEvolver|campaign|Codex update|mission_debug|'
            'run_campaign|worker_deepseek|guard convergence" /root/scratch',
            "grep -rn 'foo|codex exec|bar' src/",
            'rg "a|claude|b" /app',
            "rg -e 'droid mission run|factory mission run' logs/",
        ):
            assert external_agent_command_reason(command) is None, command
            result = ShellTool().execute(command)
            assert result.metadata.get("blocked_by") != "nested_sub_agent_creation_guard"

    def test_still_blocks_real_pipeline_into_agent_after_search(self):
        # A genuine pipeline that pipes into an agent CLI must still be blocked,
        # even when the left-hand side is a search command.
        command = 'rg -n "pattern" src/ | codex exec "fix"'
        assert external_agent_command_reason(command)

    def test_allows_workspace_tests_commands(self, tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_example.py").write_text("def test_example():\n    assert True\n")
        tool = ShellTool()
        result = tool.execute(f"ls {tmp_path / 'tests'}")
        assert result.success is True
        assert "test_example.py" in result.output


class TestFileReadTool:
    def test_read_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3")
        tool = FileReadTool()
        result = tool.execute(str(f))
        assert result.success is True
        assert "line1" in result.output
        assert "line2" in result.output

    def test_read_nonexistent(self):
        tool = FileReadTool()
        result = tool.execute("/nonexistent/file.txt")
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_read_with_offset(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("a\nb\nc\nd\ne\n")
        tool = FileReadTool()
        result = tool.execute(str(f), offset=3)
        assert "c" in result.output
        assert "a" not in result.output.split("\n")[0]

    def test_blocks_hidden_tests_path(self):
        tool = FileReadTool()
        result = tool.execute("/tests/test.sh")
        assert result.success is False
        assert result.metadata["blocked_by"] == "leaderboard_integrity_guard"

    def test_blocks_host_hl_memory_read(self):
        tool = FileReadTool()
        result = tool.execute("/host/trials/runs/old/result.json")

        assert result.success is False
        assert result.metadata["blocked_by"] == "host_memory_guard"
        assert result.metadata["blocked_reason"] == "host_memory_search"

    def test_allows_workspace_tests_path(self, tmp_path):
        f = tmp_path / "tests" / "test_example.py"
        f.parent.mkdir()
        f.write_text("def test_example():\n    assert True\n")
        tool = FileReadTool()
        result = tool.execute(str(f))
        assert result.success is True
        assert "test_example" in result.output


def test_grep_blocks_host_hl_memory_path():
    tool = GrepTool()
    result = tool.execute("Traceback", path="trials/runs")

    assert result.success is False
    assert result.metadata["blocked_by"] == "host_memory_guard"
    assert result.metadata["blocked_reason"] == "host_memory_search"


def test_glob_blocks_host_hl_memory_artifact_search():
    tool = GlobTool()
    result = tool.execute("**/trajectory.jsonl", path="/mnt/c/tmp")

    assert result.success is False
    assert result.metadata["blocked_by"] == "host_memory_guard"
    assert result.metadata["blocked_reason"] == "host_memory_search"


class TestFileEditTool:
    def test_simple_edit(self, tmp_path):
        f = tmp_path / "edit.py"
        f.write_text("x = 1\ny = 2\n")
        tool = FileEditTool()
        result = tool.execute(str(f), "x = 1", "x = 99")
        assert result.success is True

    def test_not_found(self, tmp_path):
        f = tmp_path / "edit.py"
        f.write_text("hello")
        tool = FileEditTool()
        result = tool.execute(str(f), "not in file", "replacement")
        assert result.success is False

    def test_duplicate_without_replace_all(self, tmp_path):
        f = tmp_path / "edit.py"
        f.write_text("dup\ndup\n")
        tool = FileEditTool()
        result = tool.execute(str(f), "dup", "new")
        assert result.success is False
        assert "2 times" in result.error

    def test_blocks_edit_that_stages_dependency_script(self, tmp_path):
        f = tmp_path / "download.py"
        f.write_text("print('safe')\n")
        tool = FileEditTool()
        replacement = (
            "import ssl, urllib.request\n"
            "ctx = ssl._create_unverified_context()\n"
            "urllib.request.urlopen('https://files.pythonhosted.org/pkg.whl', context=ctx)\n"
        )
        result = tool.execute(str(f), "print('safe')", replacement)

        assert result.success is False
        assert result.metadata["blocked_by"] == "staged_dependency_script_guard"
        assert f.read_text() == "print('safe')\n"

    def test_blocks_edit_that_stages_manual_deb_toolchain_chase(self, tmp_path):
        f = tmp_path / "install_toolchain.sh"
        f.write_text("#!/bin/sh\necho safe\n")
        tool = FileEditTool()
        replacement = (
            "#!/bin/sh\n"
            "dpkg -i /tmp/libstdc++-12-dev_12.2.0-14+deb12u1_amd64.deb\n"
        )
        result = tool.execute(str(f), "#!/bin/sh\necho safe", replacement)

        assert result.success is False
        assert result.metadata["blocked_by"] == "staged_dependency_script_guard"
        assert "large compiler/cross-toolchain" in result.error
        assert f.read_text() == "#!/bin/sh\necho safe\n"

    def test_blocks_edit_that_stages_nested_agent_script(self, tmp_path):
        f = tmp_path / "delegate.py"
        f.write_text("print('safe')\n")
        tool = FileEditTool()
        replacement = (
            "import subprocess\n"
            "cmd = ['codex', 'exec', 'fix']\n"
            "subprocess.run(cmd, check=True)\n"
        )
        result = tool.execute(str(f), "print('safe')", replacement)

        assert result.success is False
        assert result.metadata["blocked_by"] == "staged_dependency_script_guard"
        assert "only the master HL orchestrator may create sub-agents" in result.error
        assert f.read_text() == "print('safe')\n"

    def test_blocks_edit_that_exceeds_gpt2_codegolf_size_cap(self, tmp_path):
        f = tmp_path / "app" / "gpt2.c"
        f.parent.mkdir()
        original = "int main(void){return 0;}\n"
        f.write_text(original)
        tool = FileEditTool()
        oversized_replacement = "/*" + ("x" * 5000) + "*/ return 0;"

        result = tool.execute(str(f), "return 0;", oversized_replacement)

        assert result.success is False
        assert result.metadata["blocked_by"] == "deliverable_size_cap_write_guard"
        _assert_policy_guard_is_non_terminal(result.metadata)
        assert result.metadata["limit_bytes"] == 5000
        assert "under 5000 bytes" in result.error
        assert f.read_text() == original

    def test_blocks_host_hl_memory_edit_before_file_access(self):
        tool = FileEditTool()
        result = tool.execute("/host/trials/runs/old/result.json", "old", "new")

        assert result.success is False
        assert result.metadata["blocked_by"] == "host_memory_guard"
        assert result.metadata["blocked_reason"] == "host_memory_search"
        assert "Same-task memory summaries" in result.error


class TestFileWriteTool:
    def test_write_new_file(self, tmp_path):
        f = tmp_path / "new.txt"
        tool = FileWriteTool()
        result = tool.execute(str(f), "hello world")
        assert result.success is True
        assert f.read_text() == "hello world"

    def test_overwrite_existing(self, tmp_path):
        f = tmp_path / "exist.txt"
        f.write_text("old")
        tool = FileWriteTool()
        result = tool.execute(str(f), "new")
        assert result.success is True
        assert f.read_text() == "new"

    def test_blocks_agents_md_write(self, tmp_path):
        tool = FileWriteTool()
        result = tool.execute(str(tmp_path / "AGENTS.md"), "external solution")
        assert result.success is False
        assert result.metadata["blocked_by"] == "leaderboard_integrity_guard"

    def test_blocks_host_hl_memory_write_before_file_access(self):
        tool = FileWriteTool()
        result = tool.execute("/host/trials/runs/old/result.json", "{}")

        assert result.success is False
        assert result.metadata["blocked_by"] == "host_memory_guard"
        assert result.metadata["blocked_reason"] == "host_memory_search"
        assert "Same-task memory summaries" in result.error

    def test_blocks_staged_dependency_script_write(self, tmp_path):
        content = (
            "#!/usr/bin/env python3\n"
            "import ssl\n"
            "import urllib.request\n"
            "ctx = ssl.create_default_context()\n"
            "ctx.check_hostname = False\n"
            "ctx.verify_mode = ssl.CERT_NONE\n"
            "url = 'https://pypi.org/pypi/httpstan/4.13.0/json'\n"
            "urllib.request.urlopen(url, context=ctx, timeout=30)\n"
        )
        tool = FileWriteTool()
        path = tmp_path / "download_httpstan.py"
        result = tool.execute(str(path), content)

        assert result.success is False
        assert result.metadata["blocked_by"] == "staged_dependency_script_guard"
        _assert_policy_guard_is_non_terminal(result.metadata)
        assert not path.exists()

    def test_blocks_staged_dependency_script_write_with_package_mirror(self, tmp_path):
        content = (
            "#!/usr/bin/env python3\n"
            "import ssl\n"
            "import urllib.request\n"
            "ctx = ssl._create_unverified_context()\n"
            "url = 'https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/httpstan/'\n"
            "urllib.request.urlopen(url, context=ctx, timeout=30)\n"
        )
        tool = FileWriteTool()
        path = tmp_path / "download_httpstan_from_mirror.py"
        result = tool.execute(str(path), content)

        assert result.success is False
        assert result.metadata["blocked_by"] == "staged_dependency_script_guard"
        assert "hand-written dependency downloads" in result.error
        assert not path.exists()

    def test_blocks_staged_heavy_scientific_dependency_script_write(self, tmp_path):
        content = (
            "#!/usr/bin/env python3\n"
            "import subprocess\n"
            "subprocess.run(['python3', '-m', 'pip', 'install', 'httpstan'], check=True)\n"
        )
        tool = FileWriteTool()
        path = tmp_path / "install_httpstan.py"
        result = tool.execute(str(path), content)

        assert result.success is False
        assert result.metadata["blocked_by"] == "staged_dependency_script_guard"
        assert not path.exists()

    def test_blocks_staged_rstan_source_install_script_write(self, tmp_path):
        content = "#!/bin/sh\nR CMD INSTALL /tmp/rstan_2.32.7.tar.gz\n"
        tool = FileWriteTool()
        path = tmp_path / "install_rstan.sh"
        result = tool.execute(str(path), content)

        assert result.success is False
        assert result.metadata["blocked_by"] == "staged_dependency_script_guard"
        assert "heavy scientific/ML dependency installs" in result.error
        assert not path.exists()

    def test_blocks_staged_large_toolchain_script_write(self, tmp_path):
        content = (
            "#!/usr/bin/env python3\n"
            "import subprocess\n"
            "subprocess.run(['apt-get', 'install', '-y', 'clang', 'g++'], check=True)\n"
        )
        tool = FileWriteTool()
        path = tmp_path / "install_toolchain.py"
        result = tool.execute(str(path), content)

        assert result.success is False
        assert result.metadata["blocked_by"] == "staged_dependency_script_guard"
        assert "large compiler/cross-toolchain" in result.error
        assert not path.exists()

    def test_blocks_staged_nested_agent_script_write(self, tmp_path):
        content = "#!/bin/sh\nopenai-codex exec --json 'fix the task'\n"
        tool = FileWriteTool()
        path = tmp_path / "run_agent.sh"

        result = tool.execute(str(path), content)

        assert result.success is False
        assert result.metadata["blocked_by"] == "staged_dependency_script_guard"
        _assert_policy_guard_is_non_terminal(result.metadata)
        assert "only the master HL orchestrator may create sub-agents" in result.error
        assert not path.exists()

    def test_blocks_oversized_gpt2_codegolf_deliverable_write(self, tmp_path):
        tool = FileWriteTool()
        path = tmp_path / "app" / "gpt2.c"

        result = tool.execute(str(path), "x" * 5000)

        assert result.success is False
        assert result.metadata["blocked_by"] == "deliverable_size_cap_write_guard"
        _assert_policy_guard_is_non_terminal(result.metadata)
        assert result.metadata["content_bytes"] == 5000
        assert result.metadata["limit_bytes"] == 5000
        assert "under 5000 bytes" in result.error
        assert not path.exists()

    def test_allows_non_app_gpt2_scratch_write_over_codegolf_cap(self, tmp_path):
        tool = FileWriteTool()
        path = tmp_path / "scratch" / "gpt2.c"

        result = tool.execute(str(path), "x" * 6000)

        assert result.success is True
        assert path.read_text() == "x" * 6000

    @pytest.mark.parametrize(
        "content",
        [
            "#!/bin/sh\ncodex --help\n",
            "#!/bin/sh\nopenai-codex exec --json fix\n",
            "#!/bin/sh\nbash -lc 'codex exec --json fix'\n",
            "#!/bin/sh\nexec codex exec --json fix\n",
            "#!/bin/sh\nnohup codex exec --json fix\n",
            (
                "#!/usr/bin/env python3\n"
                "import subprocess\n"
                "subprocess.run(['python', '-m', 'codex.cli', 'exec', 'fix'], check=True)\n"
            ),
            (
                "#!/usr/bin/env python3\n"
                "import runpy\n"
                "runpy.run_module('codex.cli', run_name='__main__')\n"
            ),
            (
                "#!/usr/bin/env python3\n"
                "from runpy import run_module\n"
                "run_module('openai.codex', run_name='__main__')\n"
            ),
            (
                "#!/usr/bin/env python3\n"
                "import importlib\n"
                "importlib.import_module('codex.cli').main()\n"
            ),
            (
                "#!/usr/bin/env python3\n"
                "import subprocess\n"
                "subprocess.run(['claude-code', '--print', 'fix'], check=True)\n"
            ),
            (
                "#!/usr/bin/env python3\n"
                "import subprocess\n"
                "getattr(subprocess, 'run')(['codex', 'exec', 'fix'], check=True)\n"
            ),
            (
                "#!/usr/bin/env python3\n"
                "__import__('subprocess').run(['opencode', 'run', 'fix'], check=True)\n"
            ),
            (
                "#!/usr/bin/env python3\n"
                "import subprocess\n"
                "rr = subprocess.run\n"
                "rr(['codex', 'exec', 'fix'], check=True)\n"
            ),
            (
                "#!/usr/bin/env python3\n"
                "import subprocess\n"
                "cmd = ['codex', 'exec', 'fix']\n"
                "subprocess.run(cmd, check=True)\n"
            ),
            (
                "#!/usr/bin/env python3\n"
                "from importlib import import_module\n"
                "import_module('subprocess').run(['opencode', 'run', 'fix'], check=True)\n"
            ),
            (
                "#!/usr/bin/env python3\n"
                "import subprocess as sp\n"
                "sp.run(['codex', 'exec', 'fix'], check=True)\n"
            ),
            (
                "#!/usr/bin/env python3\n"
                "from subprocess import run as rr\n"
                "rr(['opencode', 'run', 'fix'], check=True)\n"
            ),
            "#!/bin/sh\nnpx --yes codex exec 'fix the task'\n",
            "#!/bin/sh\nuvx openai-codex exec 'fix the task'\n",
            "#!/bin/sh\npipx run codex exec 'fix the task'\n",
            "#!/bin/sh\npipx run openai-codex exec 'fix the task'\n",
            "#!/bin/sh\nuv tool run codex exec 'fix the task'\n",
            "#!/bin/sh\npoetry run codex exec 'fix the task'\n",
            "#!/bin/sh\nbun x codex exec 'fix the task'\n",
            "#!/bin/sh\nf(){ codex exec fix; }; f\n",
            "#!/bin/sh\nalias c='codex exec'; c fix\n",
            "#!/bin/sh\nc=codex; $c exec fix\n",
            "#!/bin/sh\nexport c=codex; $c exec fix\n",
            "#!/bin/sh\ndeclare c=codex; $c exec fix\n",
            "#!/bin/sh\nc=$(echo codex); $c exec fix\n",
            "#!/bin/sh\nenv c=codex bash -lc '$c exec fix'\n",
            (
                "#!/usr/bin/env python3\n"
                "import subprocess\n"
                "cmd = 'cod' + 'ex'\n"
                "subprocess.run([cmd, 'exec', 'fix'], check=True)\n"
            ),
            (
                "#!/usr/bin/env python3\n"
                "import subprocess\n"
                "subprocess.run(['co' 'dex', 'exec', 'fix'], check=True)\n"
            ),
            (
                "#!/usr/bin/env python3\n"
                "import subprocess\n"
                "cmd = ''.join(['co', 'dex'])\n"
                "subprocess.run([cmd, 'exec', 'fix'], check=True)\n"
            ),
            (
                "#!/usr/bin/env python3\n"
                "import subprocess\n"
                "cmd = f'codex'\n"
                "subprocess.run([cmd, 'exec', 'fix'], check=True)\n"
            ),
            (
                "#!/usr/bin/env python3\n"
                "import subprocess\n"
                "cmd = chr(99)+chr(111)+chr(100)+chr(101)+chr(120)\n"
                "subprocess.run([cmd, 'exec', 'fix'], check=True)\n"
            ),
            (
                "#!/usr/bin/env python3\n"
                "import subprocess\n"
                "cmd = bytes([99,111,100,101,120]).decode()\n"
                "subprocess.run([cmd, 'exec', 'fix'], check=True)\n"
            ),
            (
                "#!/usr/bin/env python3\n"
                "import subprocess\n"
                "cmd = ('codex').replace('x','x')\n"
                "subprocess.run([cmd, 'exec', 'fix'], check=True)\n"
            ),
            (
                "#!/usr/bin/env python3\n"
                "import subprocess\n"
                "cmd = 'factory'\n"
                "subprocess.run([cmd, 'mission', 'run'], check=True)\n"
            ),
            (
                "#!/usr/bin/env python3\n"
                "import subprocess\n"
                "cmd = 'droid'\n"
                "subprocess.run([cmd, 'mission', 'run'], check=True)\n"
            ),
            "#!/usr/bin/env node\nrequire('child_process').spawn('codex', ['exec', 'fix'])\n",
            "#!/usr/bin/env node\nrequire('child_process').spawn('openai-codex', ['exec', 'fix'])\n",
            "#!/usr/bin/env node\nrequire('child_process').spawn('factory', ['mission', 'run'])\n",
            "#!/usr/bin/env node\nrequire('child_process').spawn('droid', ['mission', 'run'])\n",
            "#!/usr/bin/env node\nconst cp = require('child_process'); cp.spawn('codex', ['exec', 'fix'])\n",
            "#!/usr/bin/env node\nconst cp = require('child_process'); cp.spawn('factory', ['mission', 'run'])\n",
            "#!/usr/bin/env node\nconst {spawn: s} = require('child_process'); s('opencode', ['run', 'fix'])\n",
            "#!/usr/bin/env node\nconst {spawn: s} = require('child_process'); s('droid', ['mission', 'run'])\n",
            "#!/usr/bin/env node\nconst c = 'cod' + 'ex'; require('child_process').spawn(c, ['exec', 'fix'])\n",
            "#!/usr/bin/env node\nconst c = 'factory'; require('child_process').spawn(c, ['mission', 'run'])\n",
            "#!/usr/bin/env node\nconst c = ['co','dex'].join(''); require('child_process').spawn(c, ['exec', 'fix'])\n",
            "#!/usr/bin/env node\nconst c = String.fromCharCode(99,111,100,101,120); require('child_process').spawn(c, ['exec', 'fix'])\n",
            "#!/usr/bin/env node\nconst c = Buffer.from([99,111,100,101,120]).toString(); require('child_process').spawn(c, ['exec', 'fix'])\n",
            "#!/usr/bin/env node\nconst cp = require('child_process'); cp.execSync('codex exec fix')\n",
            "#!/usr/bin/env node\nimport {spawnSync} from 'child_process'; spawnSync('codex', ['exec', 'fix'])\n",
            "#!/usr/bin/env node\nimport {spawn as s} from 'node:child_process'; s('opencode', ['run', 'fix'])\n",
            "#!/usr/bin/env node\nimport * as cp from 'node:child_process'; cp.execSync('codex exec fix')\n",
            "#!/usr/bin/env ruby\nsystem('opencode run fix')\n",
            "#!/usr/bin/env ruby\nsystem 'opencode run fix'\n",
            "#!/usr/bin/env ruby\nspawn 'codex exec fix'\n",
            "#!/usr/bin/env ruby\nspawn 'openai-codex exec fix'\n",
            "#!/usr/bin/env ruby\nspawn 'factory mission run'\n",
            "#!/usr/bin/env ruby\nspawn 'droid mission run'\n",
            "#!/usr/bin/env ruby\nc='cod'+'ex'\nspawn c, 'exec', 'fix'\n",
            "#!/usr/bin/env ruby\nc='droid'\nspawn c, 'mission', 'run'\n",
            "#!/usr/bin/env ruby\nc=['co','dex'].join\nspawn c, 'exec', 'fix'\n",
            "#!/usr/bin/env ruby\nc=%q{codex}\nspawn c, 'exec', 'fix'\n",
            "#!/usr/bin/env ruby\nsend(:system, 'codex exec fix')\n",
            "#!/usr/bin/env lua\nos.execute('codex exec fix')\n",
            "#!/usr/bin/env lua\nio.popen('codex exec fix')\n",
            "<?php system('codex exec fix'); ?>\n",
            "<?php exec('codex exec fix'); ?>\n",
            "run:\n\tfind . -exec codex exec fix ;\n",
            "run:\n\tscript -q -c 'codex exec fix' /dev/null\n",
            "run:\n\teval 'bash -lc \\\"codex exec fix\\\"'\n",
        ],
    )
    def test_blocks_staged_nested_agent_script_write_through_wrappers(self, tmp_path, content):
        tool = FileWriteTool()
        path = tmp_path / "Makefile" if content.startswith("run:") else tmp_path / "run_wrapped_agent.sh"

        result = tool.execute(str(path), content)

        assert result.success is False
        assert result.metadata["blocked_by"] == "staged_dependency_script_guard"
        _assert_policy_guard_is_non_terminal(result.metadata)
        assert "only the master HL orchestrator may create sub-agents" in result.error
        assert not path.exists()

    def test_allows_safe_script_write_with_task_data_url(self, tmp_path):
        content = (
            "#!/usr/bin/env python3\n"
            "import urllib.request\n"
            "urllib.request.urlretrieve('https://example.com/task-data.tar.gz', '/tmp/data.tar.gz')\n"
        )
        tool = FileWriteTool()
        path = tmp_path / "fetch_task_data.py"
        result = tool.execute(str(path), content)

        assert result.success is True
        assert path.read_text() == content


class TestGrepTool:
    def test_basic_search(self, tmp_path):
        f = tmp_path / "search.py"
        f.write_text("def foo():\n    pass\ndef bar():\n    pass\n")
        tool = GrepTool()
        result = tool.execute("def foo", str(tmp_path))
        assert result.success is True
        assert "foo" in result.output

    def test_no_match(self, tmp_path):
        f = tmp_path / "empty.py"
        f.write_text("nothing here")
        tool = GrepTool()
        result = tool.execute("xyzzy_nonexistent", str(tmp_path))
        assert "(no matches)" in result.output


class TestGlobTool:
    def test_find_py_files(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        tool = GlobTool()
        result = tool.execute("*.py", str(tmp_path))
        assert "a.py" in result.output
        assert "b.py" in result.output
        assert "c.txt" not in result.output
