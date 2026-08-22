"""Standardized local verification tool."""

from __future__ import annotations

import subprocess
import time
import re
from dataclasses import dataclass, field
from typing import Any

from harness.tools.base import (
    ToolDef,
    ToolResult,
    ToolSchema,
    operation_timeout_metadata,
    policy_guard_metadata,
)
from harness.tools.shell import external_agent_command_reason, shell_semantic_failure_kind


_ONLY_ACHIEVED_THRESHOLD = re.compile(
    r"\bonly achieved\s+(?P<actual>\d+(?:\.\d+)?)\s*%?"
    r"(?:(?!\n).){0,200}?\(\s*need\s+"
    r"(?P<need>\d+(?:\.\d+)?)\s*%?\+?\s*\)",
    re.IGNORECASE,
)
_EXPLICIT_THRESHOLD_FAIL = re.compile(
    r"(?mi)^\s*.{0,180}\bFAIL\b.{0,120}\(\s*need\s+[^)]{1,120}\)"
)
_FAILED_COUNT_SUMMARY = re.compile(
    r"(?mi)^\s*FAILED\s+\(\s*(?P<count>[1-9]\d*)\s*/\s*\d+\s*\)\s*:"
    r"\s*(?!None\s*$).+"
)
_COREWAR_WIN_LINE = re.compile(
    r"(?mi)^\s*(?P<opponent>stone|paper|vampire|snake|g2-clear)(?:\.red)?\s*:"
    r"\s*(?P<wins>\d+)\s+wins\s*\(\s*Results:\s*\d+\s+\d+\s+\d+\s*\)"
)
_COREWAR_THRESHOLDS = {
    "stone": 75,
    "paper": 75,
    "vampire": 75,
    "snake": 33,
    "g2-clear": 33,
}
_REGEX_BACKREFERENCE_ERROR = re.compile(
    r"(?:re\.PatternError:\s*)?invalid group reference\s+\d+",
    re.IGNORECASE,
)
_REGEX_BACKREFERENCE_CONTEXT = re.compile(
    r"\b(?:re\.sub|parse_template|pattern\.sub|replacement|re\.json|PatternError)\b",
    re.IGNORECASE,
)
_PLAIN_LOG_INSPECTION_COMMAND = re.compile(
    r"^\s*(?:cat|grep|egrep|fgrep|rg|sed|awk|head|tail|less|more|wc)\b",
    re.IGNORECASE,
)


def verify_semantic_failure_kind(
    output: str,
    command: str = "",
    returncode: int | None = None,
) -> str | None:
    """Detect verifier failures hidden behind a successful process status."""
    shell_failure = shell_semantic_failure_kind(
        output,
        command=command,
        returncode=returncode,
    )
    if shell_failure:
        return shell_failure
    if _threshold_assertion_failure(output):
        return "verification_threshold_failure"
    if _corewar_threshold_failure(output):
        return "corewar_threshold_failure"
    if _regex_replacement_backreference_failure(output, command):
        return "regex_replacement_backreference_failure"
    return None


def verify_semantic_failure_error(kind: str, returncode: int) -> str:
    prefix = f"exit code: {returncode}; " if returncode != 0 else ""
    if kind == "large_graphics_runtime_install_plan":
        return prefix + (
            "package manager output indicates a large graphics/CV runtime "
            "install plan; stop this Mesa/OpenGL/Vulkan/OpenCV path and pivot "
            "to an existing lightweight image dependency, Python stdlib parsing, "
            "or a dependency-light artifact that satisfies the visible output contract."
        )
    if kind == "large_toolchain_install_plan":
        return prefix + (
            "package manager output indicates a large compiler/toolchain "
            "install plan; stop this install path and pivot to an existing "
            "toolchain, smaller build target, or dependency-free implementation."
        )
    if kind == "large_package_install_plan":
        return prefix + (
            "package manager output indicates a large transitive package "
            "install plan; stop this dependency-expansion path and pivot to "
            "an existing installed/cached artifact, smaller explicit "
            "requirement, or dependency-free implementation plus a short "
            "visible check."
        )
    if kind == "package_manager_failure":
        return prefix + (
            "package manager output indicates failure despite shell exit status; "
            "inspect stdout/stderr and switch recovery strategy."
        )
    if kind == "heavy_ml_cv_import_failure":
        return prefix + (
            "heavy ML/CV import output indicates missing native or model "
            "dependencies despite shell exit status; keep torch, mobile_sam, "
            "cv2, PIL/Pillow, numpy, and pandas behind optional/lazy imports, "
            "then pivot to dependency-light artifact and CSV/image contract "
            "checks before more package installation."
        )
    if kind == "numpy_eigensolver_failure":
        return prefix + (
            "NumPy eigensolver output indicates complex dtype handling failed "
            "despite shell exit status; repair eigen.py using available NumPy, "
            "handle complex eigenpairs without float64 in-place subtraction, "
            "normalize eigenvectors, check residuals, and run small diagonal, "
            "random, and eval.py checks before any SciPy/compiler install path."
        )
    if kind == "numpy_eigensolver_speed_threshold_failure":
        return prefix + (
            "NumPy eigensolver verifier output indicates the candidate is slower "
            "than the reference despite shell exit status; optimize eigen.py for "
            "the visible small float64 matrix sizes, preserve correct eigenpair "
            "normalization/residual checks, benchmark sizes 2-10 with the task's "
            "timing harness, and avoid SciPy/compiler dependency paths unless "
            "already available."
        )
    if kind == "single_file_deliverable_directory_contract":
        return prefix + (
            "verifier output indicates a single-file deliverable directory "
            "contract failure despite shell exit status; create /app/polyglot "
            "early when required, keep exactly the expected main.rs or main.py.c "
            "in that final directory, move scratch probes, compiled binaries, "
            "object files, and test_* artifacts outside it, then rerun the "
            "os.listdir exact-file-list check and visible compiler/interpreter "
            "commands against the final file only."
        )
    if kind == "gpt2_codegolf_text_contract":
        return prefix + (
            "verifier output indicates the GPT2 codegolf text contract failed "
            "despite shell exit status; preserve /app/gpt2.c, keep it under "
            "5000 bytes, compile with gcc -O3 /app/gpt2.c -lm, then run "
            "/app/a.out gpt2-124M.ckpt vocab.bpe 'THIS SOFTWARE IS PROVIDED "
            "\"AS IS\", WITHOUT' and verify stdout contains WARRANTY OF ANY "
            "KIND, EXPRESS OR IMPLIED as valid UTF-8 continuation text rather "
            "than repeated token ids, escaped binary garbage, prompt-only output, "
            "or a 90s timeout."
        )
    if kind == "structured_csv_table_contract":
        return prefix + (
            "verifier output indicates a structured CSV/table contract failure "
            "despite shell exit status; preserve the exact CSV output path and "
            "schema, then run one focused readback check with pandas pd.read_csv "
            "or stdlib csv.DictReader/csv.reader that verifies header/column "
            "order, exact row count, key or identifier values, "
            "blank-vs-nonblank cells, numeric/text formatting, and expected "
            "keyed row content before broad document, image, data parsing, or "
            "package expansion."
        )
    if kind == "missing_output_artifact_contract":
        return prefix + (
            "verifier output indicates a missing output artifact contract "
            "failure despite shell exit status; create or repair the exact "
            "verifier-named /app artifact path first, then run one tiny "
            "existence or shape check such as test -s, Path(...).exists(), "
            "file, head, wc, json.load, csv reader, or a format-specific parser "
            "before broad solver rewrites, package installation, full validation, "
            "or artifact-wide searches."
        )
    if kind == "dna_insert_primer_pair_contract":
        return prefix + (
            "verifier output indicates DNA insert primer-pair contract failure "
            "despite shell exit status; preserve /app/primers.fasta and run one "
            "focused parser checking exactly 4 FASTA lines, ATCG-only "
            "forward/reverse primers, primers_concat = rc(rev_primer) + "
            "fwd_primer, inserted DNA placement, annealed overlaps 15..45 nt, "
            "vector suffix/prefix matches, Tm 58..72 C, and forward/reverse "
            "Tm delta <=5 before primer3/toolchain/package expansion."
        )
    if kind == "dna_assembly_primer_contract":
        return prefix + (
            "verifier output indicates DNA assembly primer contract failure "
            "despite shell exit status; preserve /app/primers.fasta and run one "
            "focused parser/checker for exact required two-line FASTA entries "
            "and headers, ATCG-only sequences, at least one clamp before "
            "ggtctc/BsaI, the four-base overhang immediately after the BsaI "
            "site, and parse_bsai_primer/make_fragment semantics before broad "
            "primer redesign or dependency setup."
        )
    if kind in {"verification_threshold_failure", "corewar_threshold_failure"}:
        return prefix + (
            "verification output indicates an unmet threshold despite shell exit "
            "status; inspect stdout/stderr and repair before completion."
        )
    if kind == "regex_replacement_backreference_failure":
        return prefix + (
            "verification output indicates an invalid Python re.sub replacement "
            "backreference despite shell exit status; inspect pattern/replacement "
            "pairs, use explicit \\g<N> references where needed, and repair before "
            "completion."
        )
    if kind == "masked_build_test_failure":
        return (
            "build/test output indicates failure despite shell exit status; "
            "inspect stdout/stderr and repair before completion."
        )
    if kind == "network_probe_tool_missing":
        return (
            "network probe output indicates the requested probe tool is missing "
            "despite shell exit status; treat this as reachability-probe evidence, "
            "use an installed probe such as Python urllib when needed, and do not "
            "repeat the same missing-tool command."
        )
    return prefix + (
        "verification output indicates failure despite shell exit status; "
        "inspect stdout/stderr and switch recovery strategy."
    )


def _threshold_assertion_failure(output: str) -> bool:
    if not output.strip():
        return False
    for match in _ONLY_ACHIEVED_THRESHOLD.finditer(output):
        if float(match.group("actual")) < float(match.group("need")):
            return True
    return bool(
        _EXPLICIT_THRESHOLD_FAIL.search(output)
        or _FAILED_COUNT_SUMMARY.search(output)
    )


def _corewar_threshold_failure(output: str) -> bool:
    for match in _COREWAR_WIN_LINE.finditer(output):
        opponent = match.group("opponent").lower()
        threshold = _COREWAR_THRESHOLDS.get(opponent)
        if threshold is not None and int(match.group("wins")) < threshold:
            return True
    return False


def _regex_replacement_backreference_failure(output: str, command: str) -> bool:
    if not output.strip() or not _REGEX_BACKREFERENCE_ERROR.search(output):
        return False
    if command.strip() and _PLAIN_LOG_INSPECTION_COMMAND.search(command):
        return False
    return bool(_REGEX_BACKREFERENCE_CONTEXT.search(output))


@dataclass
class VerifyTool(ToolDef):
    name: str = "verify"
    version: str = "0.1.0"
    dependencies: list[str] = field(default_factory=list)
    description: str = (
        "Run a bounded local verification command before declaring the task ready "
        "for Harbor verification. This does not decide benchmark pass/fail."
    )
    timeout_seconds: float = 120.0

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Verification command."},
                    "timeout": {"type": "number", "description": "Timeout in seconds."},
                },
                "required": ["command"],
            },
        )

    def execute(
        self,
        command: str,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        external_agent_reason = external_agent_command_reason(command)
        if external_agent_reason:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "Verification policy blocked command: "
                    f"{external_agent_reason}. Keep agent creation in the master "
                    "campaign orchestrator and solve the task with this Worker loop."
                ),
                metadata=policy_guard_metadata(
                    "nested_sub_agent_creation_guard",
                    sub_agent_creation_guard=True,
                    nested_sub_agent_creation_allowed=False,
                    only_master_loop_may_create_sub_agents=True,
                    sub_agent_creation_loop_stop_condition=False,
                    nested_sub_agent_creation_stop_condition=False,
                ),
            )
        timeout = timeout or self.timeout_seconds
        start = time.time()
        try:
            completed = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=kwargs.get("cwd"),
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"verification timed out after {timeout}s. This is an "
                    "operation timeout, not a Worker, sub-agent, or master "
                    "loop stop condition."
                ),
                duration_ms=(time.time() - start) * 1000,
                metadata=operation_timeout_metadata(
                    timeout_seconds=timeout,
                    requested_timeout_seconds=timeout,
                    elapsed_ms=(time.time() - start) * 1000,
                    telemetry_source="verify",
                ),
            )
        output = completed.stdout
        if completed.stderr:
            output += f"\n[stderr]\n{completed.stderr}"
        semantic_failure = verify_semantic_failure_kind(
            output,
            command=command,
            returncode=completed.returncode,
        )
        metadata = {"exit_code": completed.returncode}
        if semantic_failure:
            metadata.update(
                {
                    "semantic_failure_detected": True,
                    "semantic_failure_kind": semantic_failure,
                    "verification_semantic_failure_stop_condition": False,
                    "loop_stop_condition": False,
                }
            )
        success = completed.returncode == 0 and semantic_failure is None
        return ToolResult(
            success=success,
            output=output,
            error=(
                verify_semantic_failure_error(semantic_failure, completed.returncode)
                if semantic_failure
                else "" if completed.returncode == 0 else f"exit code: {completed.returncode}"
            ),
            duration_ms=(time.time() - start) * 1000,
            metadata=metadata,
        )
