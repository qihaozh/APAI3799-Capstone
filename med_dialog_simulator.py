"""
MedDialogSimulator - 基于中国医疗对话数据集的患者-医生对话模拟器

该模块从 Medical-Dialogue-Dataset-Chinese 数据集中解析真实的医疗对话，
并利用多种 AI 模型（Google Gemini / OpenAI ChatGPT / Anthropic Claude）
来模拟患者与医生之间的多轮对话。

支持的模型后端:
  - gemini   : Google Gemini (默认 gemini-2.0-flash)   — pip install google-genai
  - chatgpt  : OpenAI GPT 系列 (默认 gpt-4o-mini)      — pip install openai
  - claude   : Anthropic Claude (默认 claude-sonnet-4-20250514) — pip install anthropic
"""

from __future__ import annotations

import json
import os
import random
import re
import textwrap
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Generator, Literal

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DialogueTurn:
    """对话中的单轮消息。"""
    role: Literal["patient", "doctor"]  # 角色：病人 / 医生
    text: str                            # 消息内容

@dataclass
class MedicalRecord:
    """从数据集中解析出的一条完整医疗记录。"""
    record_id: int = -1
    url: str = ""
    hospital: str = ""
    department: str = ""
    disease: str = ""
    description: str = ""
    hope: str = ""
    allergy: str = ""
    medical_history: str = ""
    dialogue: list[DialogueTurn] = field(default_factory=list)
    diagnosis: str = ""
    suggestion: str = ""

    # ---- 辅助方法 ----
    def to_dict(self) -> dict:
        return asdict(self)

    def patient_profile_prompt(self) -> str:
        """将病人信息组织成可供 LLM 扮演病人的 prompt 片段。"""
        parts = [
            f"疾病/主诉: {self.disease}",
            f"病情描述: {self.description}",
        ]
        if self.allergy:
            parts.append(f"过敏史: {self.allergy}")
        if self.medical_history:
            parts.append(f"既往病史: {self.medical_history}")
        if self.hope:
            parts.append(f"希望获得的帮助: {self.hope}")
        return "\n".join(parts)

    def doctor_profile_prompt(self) -> str:
        """将医生信息组织成可供 LLM 扮演医生的 prompt 片段。"""
        parts = [
            f"医院: {self.hospital}",
            f"科室: {self.department}",
        ]
        if self.diagnosis:
            parts.append(f"参考诊断: {self.diagnosis}")
        if self.suggestion:
            parts.append(f"参考建议: {self.suggestion}")
        return "\n".join(parts)

    def reference_dialogue_text(self) -> str:
        """将真实对话格式化为可读文本。"""
        if not self.dialogue:
            return "（该记录没有对话内容）"
        lines: list[str] = []
        for turn in self.dialogue:
            role_label = "病人" if turn.role == "patient" else "医生"
            lines.append(f"{role_label}：{turn.text}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dataset parser
# ---------------------------------------------------------------------------

class DatasetParser:
    """解析 Medical-Dialogue-Dataset-Chinese 数据集的 .txt 文件。"""

    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(f"数据集目录不存在: {self.dataset_dir}")

    def available_files(self) -> list[Path]:
        """列出数据集目录下的所有 .txt 文件。"""
        return sorted(self.dataset_dir.glob("*.txt"))

    # ------------------------------------------------------------------ #
    # 主解析方法
    # ------------------------------------------------------------------ #
    def parse_file(self, filepath: str | Path, *, limit: int | None = None) -> list[MedicalRecord]:
        """
        解析单个数据文件，返回 MedicalRecord 列表。

        Parameters
        ----------
        filepath : str | Path
            数据文件路径。
        limit : int | None
            最多解析多少条记录，None 表示全部。
        """
        filepath = Path(filepath)
        records: list[MedicalRecord] = []
        current_lines: list[str] = []

        with open(filepath, "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.rstrip("\n")
                # 检测到新记录的起始行
                if re.match(r"^id=\d+", stripped) and current_lines:
                    rec = self._parse_block(current_lines)
                    if rec is not None:
                        records.append(rec)
                        if limit is not None and len(records) >= limit:
                            return records
                    current_lines = []
                current_lines.append(stripped)

            # 最后一条记录
            if current_lines:
                rec = self._parse_block(current_lines)
                if rec is not None:
                    records.append(rec)

        return records

    def iter_records(self, filepath: str | Path) -> Generator[MedicalRecord, None, None]:
        """逐条 yield 记录，适合处理大文件。"""
        filepath = Path(filepath)
        current_lines: list[str] = []

        with open(filepath, "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.rstrip("\n")
                if re.match(r"^id=\d+", stripped) and current_lines:
                    rec = self._parse_block(current_lines)
                    if rec is not None:
                        yield rec
                    current_lines = []
                current_lines.append(stripped)

            if current_lines:
                rec = self._parse_block(current_lines)
                if rec is not None:
                    yield rec

    def sample_records(
        self,
        filepath: str | Path,
        n: int = 5,
        *,
        with_dialogue: bool = True,
        seed: int | None = None,
    ) -> list[MedicalRecord]:
        """
        从文件中随机采样 n 条记录。

        Parameters
        ----------
        with_dialogue : bool
            若为 True，只采样包含对话的记录。
        """
        if seed is not None:
            random.seed(seed)

        # 先通过 reservoir sampling 收集候选记录
        candidates: list[MedicalRecord] = []
        for rec in self.iter_records(filepath):
            if with_dialogue and not rec.dialogue:
                continue
            candidates.append(rec)

        if len(candidates) <= n:
            return candidates
        return random.sample(candidates, n)

    # ------------------------------------------------------------------ #
    # 内部解析
    # ------------------------------------------------------------------ #
    def _parse_block(self, lines: list[str]) -> MedicalRecord | None:
        """解析一条记录的文本块。"""
        text = "\n".join(lines)
        rec = MedicalRecord()

        # --- id ---
        m = re.search(r"^id=(\d+)", text)
        if m:
            rec.record_id = int(m.group(1))
        else:
            return None

        # --- url ---
        m = re.search(r"(https?://\S+)", text)
        if m:
            rec.url = m.group(1)

        # --- Doctor faculty ---
        m = re.search(r"Doctor faculty\n(.+?)(?:\n\n|\nDescription)", text, re.DOTALL)
        if m:
            faculty_text = m.group(1).strip()
            parts = [p.strip() for p in faculty_text.split("  ") if p.strip()]
            if len(parts) >= 2:
                rec.hospital = parts[0]
                rec.department = parts[1]
            elif parts:
                rec.hospital = parts[0]

        # --- Description ---
        desc_match = re.search(r"Description\n(.*?)(?=\nDialogue\n|\nDiagnosis and suggestions\n|\nid=|\Z)", text, re.DOTALL)
        if desc_match:
            desc_text = desc_match.group(1)
            # 疾病
            m = re.search(r"疾病[：:]\s*\n?(.+?)(?=\n病情描述|$)", desc_text, re.DOTALL)
            if m:
                rec.disease = m.group(1).strip()
            # 病情描述
            m = re.search(r"病情描述[：:]\s*\n?(.+?)(?=\n希望获得|$)", desc_text, re.DOTALL)
            if m:
                rec.description = m.group(1).strip()
            # 希望获得的帮助
            m = re.search(r"希望获得的帮助[：:]\s*\n?(.+?)(?=\n怀孕情况|患病多久|\n用药情况|\n过敏史|\n既往病史|$)", desc_text, re.DOTALL)
            if m:
                rec.hope = m.group(1).strip()
            # 过敏史
            m = re.search(r"过敏史[：:]\s*\n?(.+?)(?=\n既往病史|$)", desc_text, re.DOTALL)
            if m:
                rec.allergy = m.group(1).strip()
            # 既往病史
            m = re.search(r"既往病史[：:]\s*\n?(.+?)(?=\n|$)", desc_text, re.DOTALL)
            if m:
                rec.medical_history = m.group(1).strip()

        # --- Dialogue ---
        dial_match = re.search(r"Dialogue\n(.*?)(?=\nDiagnosis and suggestions\n|\nid=|\Z)", text, re.DOTALL)
        if dial_match:
            dial_text = dial_match.group(1).strip()
            rec.dialogue = self._parse_dialogue(dial_text)

        # --- Diagnosis and suggestions ---
        diag_match = re.search(r"Diagnosis and suggestions\n(.*?)(?=\nid=|\Z)", text, re.DOTALL)
        if diag_match:
            diag_text = diag_match.group(1)
            m = re.search(r"病情摘要及初步印象[：:]\s*\n?(.+?)(?=\n总结建议|$)", diag_text, re.DOTALL)
            if m:
                rec.diagnosis = m.group(1).strip()
            m = re.search(r"总结建议[：:]\s*\n?(.+?)(?=$)", diag_text, re.DOTALL)
            if m:
                rec.suggestion = m.group(1).strip()

        return rec

    @staticmethod
    def _parse_dialogue(text: str) -> list[DialogueTurn]:
        """
        解析对话文本，处理 `病人：` 和 `医生：` 交替出现的情况。
        数据集中一行可能同时包含多个角色的发言（空格拼接），需要拆分。
        """
        turns: list[DialogueTurn] = []
        # 使用正则按角色标签切分
        segments = re.split(r"(病人[：:]|医生[：:])", text)
        # segments 形如 ['', '病人：', '内容...', '医生：', '内容...', ...]
        i = 1  # 跳过第一个空字符串
        while i < len(segments) - 1:
            tag = segments[i].strip().rstrip("：:")
            content = segments[i + 1].strip()
            content = re.sub(r"\s{2,}", " ", content)  # 压缩多余空白
            if content:
                role: Literal["patient", "doctor"] = "patient" if tag == "病人" else "doctor"
                turns.append(DialogueTurn(role=role, text=content))
            i += 2
        return turns


# ---------------------------------------------------------------------------
# AI Model Backends (Strategy Pattern)
# ---------------------------------------------------------------------------

class BaseModelBackend:
    """AI 模型后端的基类。"""

    def generate(self, messages: list[dict], *, temperature: float = 0.7) -> str:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


class GeminiBackend(BaseModelBackend):
    """
    Google Gemini 模型后端。

    需要安装: pip install google-genai
    需要设置环境变量 GOOGLE_API_KEY 或在构造时传入 api_key。
    """

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.0-flash"):
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "请先安装 google-genai: pip install google-genai"
            )

        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "请设置环境变量 GOOGLE_API_KEY 或在构造时传入 api_key"
            )
        self._model = model
        self._client = genai.Client(api_key=self._api_key)
        self._types = types

    @property
    def name(self) -> str:
        return f"Gemini ({self._model})"

    def generate(self, messages: list[dict], *, temperature: float = 0.7) -> str:
        """
        调用 Gemini API 生成回复。

        Parameters
        ----------
        messages : list[dict]
            消息列表，每个字典包含 'role' ('user'/'model') 和 'text' 字段。
        temperature : float
            生成温度，越高越随机。
        """
        contents = []
        for msg in messages:
            role = msg["role"]  # 'user' or 'model'
            contents.append(
                self._types.Content(
                    role=role,
                    parts=[self._types.Part.from_text(text=msg["text"])],
                )
            )

        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=self._types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=2048,
            ),
        )
        return response.text.strip()


class OpenAICompatibleBackend(BaseModelBackend):
    """
    兼容 OpenAI API 格式的后端（OpenAI / Azure OpenAI / 本地 LLM 等）。

    需要安装: pip install openai
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请先安装 openai: pip install openai")

        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model
        self._client = OpenAI(api_key=self._api_key, base_url=base_url)

    @property
    def name(self) -> str:
        return f"OpenAI-compatible ({self._model})"

    def generate(self, messages: list[dict], *, temperature: float = 0.7) -> str:
        # 转换为 OpenAI 消息格式
        oai_messages = []
        for msg in messages:
            role = "assistant" if msg["role"] == "model" else msg["role"]
            oai_messages.append({"role": role, "content": msg["text"]})

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=oai_messages,
            temperature=temperature,
            max_tokens=2048,
        )
        return resp.choices[0].message.content.strip()


class ClaudeBackend(BaseModelBackend):
    """
    Anthropic Claude 模型后端。

    需要安装: pip install anthropic
    需要设置环境变量 ANTHROPIC_API_KEY 或在构造时传入 api_key。
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError("请先安装 anthropic: pip install anthropic")

        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "请设置环境变量 ANTHROPIC_API_KEY 或在构造时传入 api_key"
            )
        self._model = model
        self._client = anthropic.Anthropic(api_key=self._api_key)

    @property
    def name(self) -> str:
        return f"Claude ({self._model})"

    def generate(self, messages: list[dict], *, temperature: float = 0.7) -> str:
        """
        调用 Claude API 生成回复。

        Parameters
        ----------
        messages : list[dict]
            消息列表，每个字典包含 'role' ('user'/'model') 和 'text' 字段。
        temperature : float
            生成温度。
        """
        # 第一条 user 消息可作为 system prompt
        system_text = ""
        api_messages = []

        for i, msg in enumerate(messages):
            role = msg["role"]
            text = msg["text"]

            if i == 0 and role == "user":
                # 将第一条作为 system 指令
                system_text = text
                continue

            # Claude API 使用 'assistant' 而不是 'model'
            if role == "model":
                role = "assistant"
            api_messages.append({"role": role, "content": text})

        # 确保至少有一条 user 消息
        if not api_messages:
            api_messages.append({"role": "user", "content": system_text})
            system_text = ""

        kwargs = dict(
            model=self._model,
            max_tokens=2048,
            temperature=temperature,
            messages=api_messages,
        )
        if system_text:
            kwargs["system"] = system_text

        response = self._client.messages.create(**kwargs)
        return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

# 支持的后端别名映射
BACKEND_ALIASES: dict[str, list[str]] = {
    "gemini":  ["gemini", "google", "google-gemini"],
    "chatgpt": ["chatgpt", "openai", "gpt"],
    "claude":  ["claude", "anthropic"],
}


def create_backend(
    backend_type: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> BaseModelBackend:
    """
    工厂函数 — 根据名称创建对应的 AI 模型后端。

    Parameters
    ----------
    backend_type : str
        后端类型名称。支持以下值（不区分大小写）：
        - 'gemini' / 'google'       → GeminiBackend
        - 'chatgpt' / 'openai' / 'gpt' → OpenAICompatibleBackend
        - 'claude' / 'anthropic'    → ClaudeBackend
    api_key : str | None
        API 密钥，也可通过对应的环境变量设置。
    model : str | None
        模型名称，None 则使用各后端的默认模型。
    base_url : str | None
        仅 OpenAI 兼容后端使用，用于自定义 API 地址。

    Returns
    -------
    BaseModelBackend

    Examples
    --------
    >>> backend = create_backend("gemini", api_key="...")
    >>> backend = create_backend("chatgpt", model="gpt-4o")
    >>> backend = create_backend("claude", model="claude-sonnet-4-20250514")
    """
    key = backend_type.strip().lower()

    # 解析别名
    resolved: str | None = None
    for canonical, aliases in BACKEND_ALIASES.items():
        if key in aliases:
            resolved = canonical
            break

    if resolved is None:
        supported = ", ".join(
            f"{k} ({'/'.join(v)})" for k, v in BACKEND_ALIASES.items()
        )
        raise ValueError(
            f"不支持的后端类型: '{backend_type}'。支持的类型: {supported}"
        )

    if resolved == "gemini":
        kwargs: dict = {"api_key": api_key}
        if model:
            kwargs["model"] = model
        return GeminiBackend(**kwargs)

    elif resolved == "chatgpt":
        kwargs = {"api_key": api_key}
        if model:
            kwargs["model"] = model
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAICompatibleBackend(**kwargs)

    else:  # claude
        kwargs = {"api_key": api_key}
        if model:
            kwargs["model"] = model
        return ClaudeBackend(**kwargs)


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------

class MedDialogSimulator:
    """
    基于中国医疗对话数据集的患者-医生对话模拟器。

    核心功能
    --------
    1. **解析数据集** — 从 Medical-Dialogue-Dataset-Chinese 中加载真实记录。
    2. **模拟对话** — 使用 Gemini / OpenAI 等 AI 模型，基于真实病例生成
       多轮患者-医生对话。
    3. **多种模式** — 支持 AI 同时扮演双方 / AI 扮演医生（用户扮演患者）
       / AI 扮演患者（用户扮演医生）等模式。

    Quick Start
    -----------
    >>> sim = MedDialogSimulator(
    ...     dataset_dir="Medical-Dialogue-Dataset-Chinese",
    ...     backend=GeminiBackend(api_key="YOUR_KEY"),
    ... )
    >>> records = sim.load_records("2020.txt", limit=10)
    >>> result = sim.simulate(records[3], max_turns=6)
    >>> print(result.formatted())
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        backend: BaseModelBackend | None = None,
    ):
        """
        Parameters
        ----------
        dataset_dir : str | Path
            Medical-Dialogue-Dataset-Chinese 数据集目录路径。
        backend : BaseModelBackend | None
            AI 模型后端实例。传 None 时只能使用解析功能，不能模拟对话。
        """
        self.parser = DatasetParser(dataset_dir)
        self.backend = backend
        self._records_cache: dict[str, list[MedicalRecord]] = {}

    # ------------------------------------------------------------------ #
    # 数据加载
    # ------------------------------------------------------------------ #
    def list_files(self) -> list[str]:
        """列出数据集中可用的文件名。"""
        return [f.name for f in self.parser.available_files()]

    def load_records(
        self,
        filename: str,
        *,
        limit: int | None = None,
        cache: bool = True,
    ) -> list[MedicalRecord]:
        """
        加载指定文件的记录。

        Parameters
        ----------
        filename : str
            文件名，例如 '2020.txt'。
        limit : int | None
            最多加载多少条。
        cache : bool
            是否缓存结果。
        """
        key = f"{filename}:{limit}"
        if cache and key in self._records_cache:
            return self._records_cache[key]

        filepath = self.parser.dataset_dir / filename
        records = self.parser.parse_file(filepath, limit=limit)

        if cache:
            self._records_cache[key] = records
        return records

    def sample_records(
        self,
        filename: str,
        n: int = 5,
        *,
        with_dialogue: bool = True,
        seed: int | None = None,
    ) -> list[MedicalRecord]:
        """从指定文件中随机采样 n 条记录。"""
        filepath = self.parser.dataset_dir / filename
        return self.parser.sample_records(
            filepath, n, with_dialogue=with_dialogue, seed=seed
        )

    def get_record(self, filename: str, record_id: int) -> MedicalRecord | None:
        """按 record_id 从文件中查找特定记录。"""
        for rec in self.parser.iter_records(self.parser.dataset_dir / filename):
            if rec.record_id == record_id:
                return rec
        return None

    # ------------------------------------------------------------------ #
    # 对话模拟
    # ------------------------------------------------------------------ #

    @dataclass
    class SimulationResult:
        """模拟对话的结果。"""
        record: MedicalRecord
        simulated_dialogue: list[DialogueTurn] = field(default_factory=list)
        mode: str = ""
        model_name: str = ""

        def formatted(self) -> str:
            """格式化输出模拟结果。"""
            sep = "=" * 60
            lines = [
                sep,
                f"📋 病例 ID: {self.record.record_id}",
                f"🏥 {self.record.hospital} - {self.record.department}",
                f"🤖 模型: {self.model_name}  |  模式: {self.mode}",
                sep,
                "",
                "【患者信息】",
                self.record.patient_profile_prompt(),
                "",
                "--- 模拟对话 ---",
            ]
            for turn in self.simulated_dialogue:
                icon = "🧑‍⚕️ 医生" if turn.role == "doctor" else "🤒 病人"
                lines.append(f"{icon}：{turn.text}")
            lines.append("")

            if self.record.dialogue:
                lines.append("--- 真实对话（参考） ---")
                lines.append(self.record.reference_dialogue_text())
                lines.append("")

            if self.record.diagnosis:
                lines.append(f"📌 参考诊断: {self.record.diagnosis}")
            if self.record.suggestion:
                lines.append(f"💡 参考建议: {self.record.suggestion}")
            lines.append(sep)
            return "\n".join(lines)

        def to_dict(self) -> dict:
            return {
                "record": self.record.to_dict(),
                "simulated_dialogue": [
                    {"role": t.role, "text": t.text}
                    for t in self.simulated_dialogue
                ],
                "mode": self.mode,
                "model_name": self.model_name,
            }

    def simulate(
        self,
        record: MedicalRecord,
        *,
        max_turns: int = 8,
        temperature: float = 0.7,
        mode: Literal["auto", "doctor", "patient"] = "auto",
    ) -> SimulationResult:
        """
        基于真实病例记录模拟多轮对话。

        Parameters
        ----------
        record : MedicalRecord
            真实医疗记录。
        max_turns : int
            最大对话轮次（一问一答算 2 轮）。
        temperature : float
            生成温度。
        mode : str
            - 'auto'   : AI 同时扮演医生和患者，自动生成完整对话。
            - 'doctor' : AI 扮演医生，需用户扮演患者（交互模式，
                         此处简化为 AI 根据患者信息自动模拟首轮后生成）。
            - 'patient': AI 扮演患者。
        """
        if self.backend is None:
            raise RuntimeError("未设置 AI 模型后端，无法进行对话模拟。")

        if mode == "auto":
            return self._simulate_auto(record, max_turns, temperature)
        elif mode == "doctor":
            return self._simulate_as_doctor(record, max_turns, temperature)
        elif mode == "patient":
            return self._simulate_as_patient(record, max_turns, temperature)
        else:
            raise ValueError(f"不支持的模式: {mode}")

    # ---- Auto 模式：AI 一次性生成完整多轮对话 ----
    def _simulate_auto(
        self, record: MedicalRecord, max_turns: int, temperature: float
    ) -> SimulationResult:
        system_prompt = textwrap.dedent(f"""\
            你是一个医疗对话模拟器。请根据以下真实病例信息，模拟一段患者与医生之间的多轮中文对话。

            要求：
            1. 对话要自然、专业、符合真实医疗场景。
            2. 医生应当耐心询问病情、给出专业建议。
            3. 患者应当根据病情描述自然地回答和提问。
            4. 共生成 {max_turns} 轮对话（一轮 = 一方说一句话）。
            5. 严格按如下格式输出，每行一句，不要加额外标记：
               病人：...
               医生：...
               病人：...
               医生：...

            【患者信息】
            {record.patient_profile_prompt()}

            【医生信息】
            {record.doctor_profile_prompt()}
        """)

        ref_hint = ""
        if record.dialogue:
            ref_hint = (
                "\n\n以下是真实对话片段作为参考风格（请不要照抄，而是模拟类似风格）：\n"
                + record.reference_dialogue_text()
            )

        messages = [
            {"role": "user", "text": system_prompt + ref_hint},
        ]

        raw = self.backend.generate(messages, temperature=temperature)
        turns = self._parse_generated_dialogue(raw)

        return self.SimulationResult(
            record=record,
            simulated_dialogue=turns,
            mode="auto",
            model_name=self.backend.name,
        )

    # ---- Doctor 模式：AI 扮演医生 ----
    def _simulate_as_doctor(
        self, record: MedicalRecord, max_turns: int, temperature: float
    ) -> SimulationResult:
        system_prompt = textwrap.dedent(f"""\
            你现在扮演一名中国的专科医生，在线上问诊平台回答患者的问题。

            你的背景信息：
            {record.doctor_profile_prompt()}

            要求：
            1. 用中文、专业但通俗易懂的语言与患者交流。
            2. 主动询问相关病史、症状细节。
            3. 给出合理的建议和初步判断。
            4. 每次只回复医生的一句话。
        """)

        dialogue_turns: list[DialogueTurn] = []
        messages = [{"role": "user", "text": system_prompt}]

        # 用真实病情描述作为患者的第一句话
        first_patient_msg = f"医生你好，{record.description}"
        if record.hope:
            first_patient_msg += f" {record.hope}"

        for turn_idx in range(max_turns):
            if turn_idx % 2 == 0:
                # 患者回合
                if turn_idx == 0:
                    patient_text = first_patient_msg
                else:
                    # 使用另一次 LLM 调用生成患者回复
                    patient_text = self._generate_patient_reply(
                        record, dialogue_turns, temperature
                    )
                dialogue_turns.append(DialogueTurn(role="patient", text=patient_text))
                messages.append({"role": "user", "text": patient_text})
            else:
                # 医生回合（AI 生成）
                doctor_reply = self.backend.generate(messages, temperature=temperature)
                doctor_reply = self._clean_role_prefix(doctor_reply, "医生")
                dialogue_turns.append(DialogueTurn(role="doctor", text=doctor_reply))
                messages.append({"role": "model", "text": doctor_reply})

        return self.SimulationResult(
            record=record,
            simulated_dialogue=dialogue_turns,
            mode="doctor",
            model_name=self.backend.name,
        )

    # ---- Patient 模式：AI 扮演患者 ----
    def _simulate_as_patient(
        self, record: MedicalRecord, max_turns: int, temperature: float
    ) -> SimulationResult:
        system_prompt = textwrap.dedent(f"""\
            你现在扮演一名患者，在线上问诊平台向医生咨询。

            你的病情信息如下（请严格基于这些信息回答，不要编造不存在的症状）：
            {record.patient_profile_prompt()}

            要求：
            1. 用中文自然口语与医生交流。
            2. 如实描述自己的症状和担忧。
            3. 每次只回复患者的一句话。
        """)

        dialogue_turns: list[DialogueTurn] = []
        messages = [{"role": "user", "text": system_prompt}]

        for turn_idx in range(max_turns):
            if turn_idx % 2 == 0:
                # 患者回合（AI 生成）
                if turn_idx == 0:
                    patient_prompt = "请以患者身份描述你的症状，向医生问好并说明来意。"
                    messages.append({"role": "user", "text": patient_prompt})

                patient_reply = self.backend.generate(messages, temperature=temperature)
                patient_reply = self._clean_role_prefix(patient_reply, "病人")
                dialogue_turns.append(DialogueTurn(role="patient", text=patient_reply))
                messages.append({"role": "model", "text": patient_reply})
            else:
                # 医生回合 — 模拟医生回复
                doctor_text = self._generate_doctor_reply(
                    record, dialogue_turns, temperature
                )
                dialogue_turns.append(DialogueTurn(role="doctor", text=doctor_text))
                messages.append({"role": "user", "text": f"（医生回复）{doctor_text}"})

        return self.SimulationResult(
            record=record,
            simulated_dialogue=dialogue_turns,
            mode="patient",
            model_name=self.backend.name,
        )

    # ------------------------------------------------------------------ #
    # 交互模式
    # ------------------------------------------------------------------ #
    def interactive_chat(
        self,
        record: MedicalRecord,
        *,
        user_role: Literal["patient", "doctor"] = "patient",
        temperature: float = 0.7,
    ) -> SimulationResult:
        """
        交互式对话 — 用户在终端中实时输入，AI 扮演另一方。

        Parameters
        ----------
        record : MedicalRecord
            病例记录。
        user_role : str
            用户扮演的角色：'patient'(病人) 或 'doctor'(医生)。
        """
        if self.backend is None:
            raise RuntimeError("未设置 AI 模型后端。")

        ai_role = "doctor" if user_role == "patient" else "patient"
        ai_label = "🧑‍⚕️ 医生" if ai_role == "doctor" else "🤒 病人"
        user_label = "🤒 病人(你)" if user_role == "patient" else "🧑‍⚕️ 医生(你)"

        if ai_role == "doctor":
            system_prompt = textwrap.dedent(f"""\
                你现在扮演一名中国的专科医生，在线问诊。
                {record.doctor_profile_prompt()}
                请用中文专业但通俗的语言回复。每次只回复一句话。
            """)
        else:
            system_prompt = textwrap.dedent(f"""\
                你现在扮演一名患者，在线问诊。
                {record.patient_profile_prompt()}
                请基于以上病情信息用中文自然口语回复。每次只回复一句话。
            """)

        print("=" * 60)
        print(f"📋 病例: {record.disease}")
        print(f"🏥 {record.hospital} - {record.department}")
        print(f"你的角色: {user_label}  |  AI 角色: {ai_label}")
        print("输入 'quit' 或 'exit' 结束对话")
        print("=" * 60)

        messages = [{"role": "user", "text": system_prompt}]
        dialogue_turns: list[DialogueTurn] = []

        while True:
            user_input = input(f"\n{user_label}：").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            dialogue_turns.append(DialogueTurn(role=user_role, text=user_input))
            messages.append({"role": "user", "text": user_input})

            ai_reply = self.backend.generate(messages, temperature=temperature)
            ai_reply = self._clean_role_prefix(
                ai_reply, "医生" if ai_role == "doctor" else "病人"
            )
            print(f"{ai_label}：{ai_reply}")

            dialogue_turns.append(DialogueTurn(role=ai_role, text=ai_reply))
            messages.append({"role": "model", "text": ai_reply})

        return self.SimulationResult(
            record=record,
            simulated_dialogue=dialogue_turns,
            mode=f"interactive-{user_role}",
            model_name=self.backend.name,
        )

    # ------------------------------------------------------------------ #
    # 批量模拟 & 导出
    # ------------------------------------------------------------------ #
    def batch_simulate(
        self,
        records: list[MedicalRecord],
        *,
        max_turns: int = 8,
        temperature: float = 0.7,
        mode: Literal["auto", "doctor", "patient"] = "auto",
        verbose: bool = True,
    ) -> list[SimulationResult]:
        """对多条记录批量模拟对话。"""
        results: list[MedDialogSimulator.SimulationResult] = []
        for i, rec in enumerate(records):
            if verbose:
                print(f"[{i + 1}/{len(records)}] 模拟 id={rec.record_id} ...")
            result = self.simulate(
                rec, max_turns=max_turns, temperature=temperature, mode=mode
            )
            results.append(result)
        return results

    @staticmethod
    def export_results(
        results: list[SimulationResult],
        output_path: str | Path,
        *,
        format: Literal["json", "txt"] = "json",
    ) -> None:
        """
        将模拟结果导出到文件。

        Parameters
        ----------
        format : str
            'json' — 结构化 JSON；'txt' — 人类可读文本。
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            data = [r.to_dict() for r in results]
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif format == "txt":
            with open(output_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(r.formatted())
                    f.write("\n\n")
        else:
            raise ValueError(f"不支持的导出格式: {format}")

        print(f"✅ 已导出 {len(results)} 条结果到 {output_path}")

    # ------------------------------------------------------------------ #
    # 辅助方法
    # ------------------------------------------------------------------ #
    def _generate_patient_reply(
        self,
        record: MedicalRecord,
        history: list[DialogueTurn],
        temperature: float,
    ) -> str:
        """基于病例信息和已有对话历史，生成患者回复。"""
        history_text = "\n".join(
            f"{'病人' if t.role == 'patient' else '医生'}：{t.text}" for t in history
        )
        prompt = textwrap.dedent(f"""\
            你扮演一名患者，正在和医生在线问诊。

            你的病情信息：
            {record.patient_profile_prompt()}

            目前的对话历史：
            {history_text}

            请以患者身份自然地回复医生的上一句话。只输出患者的一句话，不要加角色标签。
        """)
        messages = [{"role": "user", "text": prompt}]
        reply = self.backend.generate(messages, temperature=temperature)
        return self._clean_role_prefix(reply, "病人")

    def _generate_doctor_reply(
        self,
        record: MedicalRecord,
        history: list[DialogueTurn],
        temperature: float,
    ) -> str:
        """基于病例信息和已有对话历史，生成医生回复。"""
        history_text = "\n".join(
            f"{'病人' if t.role == 'patient' else '医生'}：{t.text}" for t in history
        )
        prompt = textwrap.dedent(f"""\
            你扮演一名专科医生，正在线上问诊。

            医生信息：
            {record.doctor_profile_prompt()}

            目前的对话历史：
            {history_text}

            请以医生身份专业地回复患者。只输出医生的一句话，不要加角色标签。
        """)
        messages = [{"role": "user", "text": prompt}]
        reply = self.backend.generate(messages, temperature=temperature)
        return self._clean_role_prefix(reply, "医生")

    @staticmethod
    def _parse_generated_dialogue(text: str) -> list[DialogueTurn]:
        """解析 AI 生成的对话文本。"""
        turns: list[DialogueTurn] = []
        for line in text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("病人") or line.startswith("患者"):
                content = re.sub(r"^(病人|患者)[：:]\s*", "", line)
                if content:
                    turns.append(DialogueTurn(role="patient", text=content))
            elif line.startswith("医生"):
                content = re.sub(r"^医生[：:]\s*", "", line)
                if content:
                    turns.append(DialogueTurn(role="doctor", text=content))
        return turns

    @staticmethod
    def _clean_role_prefix(text: str, role_label: str) -> str:
        """去除回复文本开头可能出现的角色标签。"""
        text = text.strip()
        text = re.sub(rf"^{role_label}[：:]\s*", "", text)
        text = re.sub(r"^(病人|患者|医生)[：:]\s*", "", text)
        return text.strip()


# ---------------------------------------------------------------------------
# 便捷入口 & CLI
# ---------------------------------------------------------------------------

def quick_demo(
    dataset_dir: str = "Medical-Dialogue-Dataset-Chinese",
    api_key: str | None = None,
    filename: str = "2020.txt",
    record_id: int | None = None,
    max_turns: int = 8,
    backend_type: str = "gemini",
    model: str | None = None,
):
    """
    快速演示函数。

    Parameters
    ----------
    dataset_dir : str
        数据集目录。
    api_key : str | None
        API 密钥（也可通过环境变量设置）。
    filename : str
        要使用的数据文件。
    record_id : int | None
        指定记录 ID，None 则随机选择。
    max_turns : int
        对话轮次。
    backend_type : str
        'gemini' / 'chatgpt' / 'claude'（以及它们的别名）。
    model : str | None
        模型名称，None 则使用默认模型。
    """
    backend = create_backend(backend_type, api_key=api_key, model=model)

    sim = MedDialogSimulator(dataset_dir=dataset_dir, backend=backend)

    # 加载 / 选取记录
    if record_id is not None:
        print(f"正在查找 id={record_id} ...")
        record = sim.get_record(filename, record_id)
        if record is None:
            print(f"❌ 未找到 id={record_id}")
            return
    else:
        print(f"从 {filename} 随机采样 1 条含对话的记录 ...")
        records = sim.sample_records(filename, n=1, seed=42)
        if not records:
            print("❌ 未找到合适的记录")
            return
        record = records[0]

    print(f"✅ 选中记录 id={record.record_id}: {record.disease}\n")

    # 模拟对话
    result = sim.simulate(record, max_turns=max_turns, mode="auto")
    print(result.formatted())

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MedDialogSimulator - 医疗对话模拟器")
    parser.add_argument("--dataset-dir", default="Medical-Dialogue-Dataset-Chinese",
                        help="数据集目录路径")
    parser.add_argument("--file", default="2020.txt", help="数据文件名")
    parser.add_argument("--record-id", type=int, default=None, help="指定记录 ID")
    parser.add_argument("--max-turns", type=int, default=8, help="最大对话轮次")
    parser.add_argument("--api-key", default=None, help="API 密钥")
    parser.add_argument("--backend", default="gemini",
                        choices=["gemini", "chatgpt", "openai", "claude"],
                        help="AI 模型后端: gemini / chatgpt / claude")
    parser.add_argument("--model", default=None,
                        help="模型名称 (如 gemini-2.0-flash, gpt-4o, claude-sonnet-4-20250514)")
    parser.add_argument("--interactive", action="store_true",
                        help="启用交互模式")
    parser.add_argument("--user-role", default="patient", choices=["patient", "doctor"],
                        help="交互模式中用户扮演的角色")
    parser.add_argument("--parse-only", action="store_true",
                        help="仅解析数据集，不进行模拟")

    args = parser.parse_args()

    if args.parse_only:
        # 仅解析并展示记录
        sim = MedDialogSimulator(dataset_dir=args.dataset_dir, backend=None)
        records = sim.load_records(args.file, limit=5)
        for rec in records:
            print(f"\n{'='*50}")
            print(f"ID: {rec.record_id}")
            print(f"医院: {rec.hospital} | 科室: {rec.department}")
            print(f"疾病: {rec.disease}")
            print(f"描述: {rec.description[:100]}...")
            if rec.dialogue:
                print(f"对话轮数: {len(rec.dialogue)}")
                print("--- 对话摘录 ---")
                for turn in rec.dialogue[:4]:
                    label = "病人" if turn.role == "patient" else "医生"
                    print(f"  {label}：{turn.text[:80]}")
    elif args.interactive:
        # 交互模式
        backend = create_backend(args.backend, api_key=args.api_key, model=args.model)

        sim = MedDialogSimulator(dataset_dir=args.dataset_dir, backend=backend)

        if args.record_id is not None:
            record = sim.get_record(args.file, args.record_id)
        else:
            records = sim.sample_records(args.file, n=1, seed=42)
            record = records[0] if records else None

        if record:
            sim.interactive_chat(record, user_role=args.user_role)
        else:
            print("❌ 未找到合适的记录")
    else:
        quick_demo(
            dataset_dir=args.dataset_dir,
            api_key=args.api_key,
            filename=args.file,
            record_id=args.record_id,
            max_turns=args.max_turns,
            backend_type=args.backend,
            model=args.model,
        )
