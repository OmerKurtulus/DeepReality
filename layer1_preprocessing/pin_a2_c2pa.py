"""
DeepReality — PIN-A2: C2PA Provenance Analysis
════════════════════════════════════════════════

Görev:
    Görsel dosyasında C2PA Content Credentials (provenance) verisi
    olup olmadığını c2pa-python kütüphanesi ile kontrol eder.
    Varsa kim üretmiş, hangi araçla, ne zaman, imza geçerli mi
    bilgisini yapısal olarak çıkarır.

PIN-A1 ile farkı:
    PIN-A1 → dosya binary byte taraması (heuristik, pattern matching)
    PIN-A2 → resmi c2pa-python kütüphanesi ile manifest parse + doğrulama

Çıktı:
    has_c2pa       : bool   — C2PA manifest bulundu mu?
    creator        : str    — İmzalayan kurum (signature_info.issuer)
    tool           : str    — Üreten araç (claim_generator / softwareAgent)
    timestamp      : str    — İmza tarihi (ISO format)
    is_ai_generated: bool   — AI üretimi kanıtı var mı?
    ai_source_type : str    — Dijital kaynak tipi (IPTC)
    actions        : list   — C2PA eylem listesi
    validation     : dict   — İmza doğrulama durumu
    ingredients    : list   — Kaynak materyal zinciri
    score          : float  — 0.0 (C2PA yok / gerçek) → 1.0 (kesin AI üretimi)

Kütüphane:
    c2pa-python >= 0.28.0 (pip install c2pa-python)
    https://github.com/contentauth/c2pa-python

Yazar: DeepReality Ekibi
Tarih: 2026-02-16
"""

import json
import traceback
from pathlib import Path

from core.base_pin import BasePin
from config.settings import C2PA_CONFIG


class PinA2C2pa(BasePin):
    """
    PIN-A2: C2PA Content Credentials Provenance Analysis

    c2pa-python Reader ile dosyadaki C2PA manifest'ini okur,
    doğrular ve yapısal bilgi çıkarır.
    """

    def __init__(self):
        super().__init__(
            pin_id="PIN-A2",
            pin_name="C2PA Provenance Analysis",
            layer=1
        )
        # Config
        self.ai_source_types = C2PA_CONFIG["ai_digital_source_types"]
        self.known_issuers = C2PA_CONFIG["known_ai_issuers"]
        self.known_agents = C2PA_CONFIG["known_ai_software_agents"]
        self.non_ai_actions = C2PA_CONFIG["non_ai_actions"]
        self.ai_actions = C2PA_CONFIG["ai_creation_actions"]
        self.edit_actions = C2PA_CONFIG["edit_actions"]

    # ════════════════════════════════════════════════════════════════
    # ANA ANALİZ
    # ════════════════════════════════════════════════════════════════

    def analyze(self, file_path: str) -> dict:
        """
        C2PA manifest okuma ve analiz pipeline'ı.

        Adımlar:
            1. c2pa.Reader ile dosyayı oku
            2. Active manifest'i bul
            3. TÜM manifest zincirini tara (parent manifest'ler dahil)
            4. claim_generator → araç bilgisi çıkar
            5. signature_info → imzalayan + tarih çıkar
            6. assertions → eylemler + dijital kaynak tipi çıkar
            7. ingredients → kaynak zinciri çıkar
            8. validation_status → imza doğrulama
            9. Tüm sinyalleri skorla

        NOT: OpenAI gibi bazı sağlayıcılar çift manifest zinciri kullanır:
            Manifest 1 (parent): c2pa.created + GPT-4o + trainedAlgorithmicMedia
            Manifest 2 (active): c2pa.opened (sadece "dosya açıldı")
        Bu yüzden TÜM manifest'leri taramak zorunludur.
        """

        # ── Adım 1: C2PA manifest oku ──
        manifest_data = self._read_c2pa_manifest(file_path)

        if not manifest_data["has_c2pa"]:
            # C2PA yok → bu PIN için veri yok, skor 0
            return {
                "results": self._build_no_c2pa_results(),
                "score": 0.0,
                "verdict": "no_data",
                "details": (
                    "C2PA Content Credentials bulunamadi. "
                    "Bu dosyada C2PA provenance verisi mevcut degil. "
                    "PIN-A2 bu gorsel icin sinyal uretemiyor."
                )
            }

        # ── Adım 2: Active manifest'ten bilgi çıkar ──
        active_manifest = manifest_data["active_manifest"]

        creator_info = self._extract_creator(active_manifest)
        tool_info = self._extract_tool(active_manifest)
        timestamp_info = self._extract_timestamp(active_manifest)
        action_analysis = self._analyze_actions(active_manifest)
        source_type_analysis = self._analyze_digital_source_type(active_manifest)
        ingredient_analysis = self._analyze_ingredients(active_manifest)
        validation_info = self._extract_validation(manifest_data)

        # ── Adım 3: Tüm manifest zincirini tara (chain enrichment) ──
        # Active manifest'te eksik kalan bilgileri parent manifest'lerden çek
        all_manifests = manifest_data.get("all_manifests", {})
        active_id = manifest_data.get("active_manifest_id")
        chain_info = self._scan_full_chain(all_manifests, active_id)

        # Zenginleştirme: eksik alanları zincirden doldur
        creator_info, tool_info, timestamp_info, action_analysis, \
            source_type_analysis = self._enrich_from_chain(
                creator_info, tool_info, timestamp_info,
                action_analysis, source_type_analysis, chain_info
            )

        # ── Adım 4: Skorlama ──
        score, score_breakdown = self._calculate_score(
            creator_info=creator_info,
            tool_info=tool_info,
            action_analysis=action_analysis,
            source_type_analysis=source_type_analysis,
            ingredient_analysis=ingredient_analysis,
            validation_info=validation_info
        )

        verdict = self._determine_verdict(score)
        details = self._generate_details(
            score, verdict, creator_info, tool_info,
            timestamp_info, action_analysis, source_type_analysis,
            validation_info
        )

        return {
            "results": {
                "has_c2pa": True,
                "creator": creator_info,
                "tool": tool_info,
                "timestamp": timestamp_info,
                "actions": action_analysis,
                "digital_source_type": source_type_analysis,
                "ingredients": ingredient_analysis,
                "validation": validation_info,
                "score_breakdown": score_breakdown,
                "manifest_count": manifest_data.get("manifest_count", 0),
                "active_manifest_id": manifest_data.get("active_manifest_id"),
                "chain_info": chain_info,
            },
            "score": score,
            "verdict": verdict,
            "details": details
        }

    # ════════════════════════════════════════════════════════════════
    # C2PA MANIFEST OKUMA
    # ════════════════════════════════════════════════════════════════

    def _read_c2pa_manifest(self, file_path: str) -> dict:
        """
        c2pa.Reader ile dosyadaki C2PA manifest'ini okur.

        Returns:
            {
                "has_c2pa": bool,
                "raw_json": dict | None,
                "active_manifest": dict | None,
                "active_manifest_id": str | None,
                "manifest_count": int,
                "read_error": str | None
            }
        """
        try:
            import c2pa

            reader = c2pa.Reader(file_path)
            raw_json_str = reader.json()
            raw_json = json.loads(raw_json_str)

            active_id = raw_json.get("active_manifest")
            manifests = raw_json.get("manifests", {})

            if not active_id or active_id not in manifests:
                return {
                    "has_c2pa": False,
                    "raw_json": raw_json,
                    "active_manifest": None,
                    "active_manifest_id": None,
                    "manifest_count": len(manifests),
                    "read_error": "Active manifest bulunamadi"
                }

            return {
                "has_c2pa": True,
                "raw_json": raw_json,
                "active_manifest": manifests[active_id],
                "active_manifest_id": active_id,
                "all_manifests": manifests,
                "manifest_count": len(manifests),
                "validation_status": raw_json.get("validation_status"),
                "read_error": None
            }

        except Exception as e:
            error_msg = str(e)
            # ManifestNotFound = normal durum, C2PA yok
            if "ManifestNotFound" in error_msg:
                return {
                    "has_c2pa": False,
                    "raw_json": None,
                    "active_manifest": None,
                    "active_manifest_id": None,
                    "manifest_count": 0,
                    "read_error": None  # Hata değil, C2PA yok
                }
            else:
                # Gerçek hata
                self.errors.append(f"C2PA okuma hatasi: {error_msg}")
                return {
                    "has_c2pa": False,
                    "raw_json": None,
                    "active_manifest": None,
                    "active_manifest_id": None,
                    "manifest_count": 0,
                    "read_error": error_msg
                }

    # ════════════════════════════════════════════════════════════════
    # BİLGİ ÇIKARMA (EXTRACTION)
    # ════════════════════════════════════════════════════════════════

    def _extract_creator(self, manifest: dict) -> dict:
        """
        İmzalayan kurum bilgisini çıkarır.

        Kaynaklar:
            - signature_info.issuer (birincil)
            - claim_generator_info (ikincil)

        Returns:
            {
                "issuer": str | None,
                "is_known_ai_issuer": bool,
                "matched_issuer_key": str | None
            }
        """
        sig_info = manifest.get("signature_info", {})
        issuer = sig_info.get("issuer")

        is_known = False
        matched_key = None

        if issuer:
            issuer_lower = issuer.lower()
            for known_name, key in self.known_issuers.items():
                if known_name.lower() in issuer_lower:
                    is_known = True
                    matched_key = key
                    break

        return {
            "issuer": issuer,
            "is_known_ai_issuer": is_known,
            "matched_issuer_key": matched_key
        }

    def _extract_tool(self, manifest: dict) -> dict:
        """
        Üreten araç bilgisini çıkarır.

        Kaynaklar:
            - claim_generator (birincil, ör: "DALL-E 3/1.0 c2pa-rs/0.33.0")
            - claim_generator_info[].name (ikincil)
            - assertions içinde softwareAgent (üçüncül)

        Returns:
            {
                "claim_generator": str | None,
                "claim_generator_parsed": str | None,
                "software_agent": str | None,
                "is_known_ai_tool": bool,
                "matched_tool": str | None
            }
        """
        claim_gen = manifest.get("claim_generator")
        claim_gen_info = manifest.get("claim_generator_info")

        # claim_generator genelde "ToolName/version sdk/version" formatında
        claim_gen_parsed = None
        if claim_gen:
            # İlk parçayı al (araç adı)
            claim_gen_parsed = claim_gen.split("/")[0].strip()

        # claim_generator_info varsa
        gen_info_name = None
        if claim_gen_info and isinstance(claim_gen_info, list):
            for info in claim_gen_info:
                if isinstance(info, dict) and "name" in info:
                    gen_info_name = info["name"]
                    break

        # Assertions'dan softwareAgent ara
        software_agent = self._find_software_agent(manifest)

        # Bilinen AI aracı mı?
        is_known = False
        matched_tool = None
        check_strings = [
            s for s in [claim_gen, claim_gen_parsed, gen_info_name, software_agent]
            if s is not None
        ]
        for check_str in check_strings:
            check_lower = check_str.lower()
            for agent in self.known_agents:
                if agent.lower() in check_lower:
                    is_known = True
                    matched_tool = agent
                    break
            if is_known:
                break

        return {
            "claim_generator": claim_gen,
            "claim_generator_parsed": claim_gen_parsed or gen_info_name,
            "software_agent": software_agent,
            "is_known_ai_tool": is_known,
            "matched_tool": matched_tool
        }

    def _find_software_agent(self, manifest: dict) -> str | None:
        """
        Assertions içinde softwareAgent alanını arar.

        İki format desteklenir:
        - c2pa.actions (v1): softwareAgent → action.parameters.softwareAgent (str)
        - c2pa.actions.v2:   softwareAgent → action.softwareAgent (dict veya str)
        """
        assertions = manifest.get("assertions", [])
        for assertion in assertions:
            label = assertion.get("label", "")
            data = assertion.get("data", {})

            if "c2pa.actions" in label:
                actions = data.get("actions", [])
                for action in actions:

                    # v2 formatı: softwareAgent direkt action içinde (dict)
                    agent = action.get("softwareAgent")
                    if agent:
                        if isinstance(agent, dict):
                            name = agent.get("name", "")
                            if name:
                                return name
                        elif isinstance(agent, str):
                            return agent

                    # v1 formatı: softwareAgent parameters içinde
                    params = action.get("parameters", {})
                    if isinstance(params, dict):
                        param_agent = params.get("softwareAgent")
                        if param_agent:
                            if isinstance(param_agent, dict):
                                return param_agent.get("name", str(param_agent))
                            return str(param_agent)

            # stds.schema-org.CreativeWork içinde de olabilir
            if "schema-org" in label and "CreativeWork" in str(label):
                authors = data.get("author", [])
                if isinstance(authors, list):
                    for author in authors:
                        if isinstance(author, dict):
                            name = author.get("name")
                            if name:
                                for known_agent in self.known_agents:
                                    if known_agent.lower() in name.lower():
                                        return name

        return None

    def _extract_timestamp(self, manifest: dict) -> dict:
        """
        İmza tarih bilgisini çıkarır.

        Kaynaklar:
            - signature_info.time (birincil)
            - claim_generator_info[].time (ikincil)

        Returns:
            {
                "signature_time": str | None,
                "has_timestamp": bool
            }
        """
        sig_info = manifest.get("signature_info", {})
        sig_time = sig_info.get("time")

        return {
            "signature_time": sig_time,
            "has_timestamp": sig_time is not None
        }

    def _analyze_actions(self, manifest: dict) -> dict:
        """
        C2PA assertion'larındaki eylemleri analiz eder.

        c2pa.actions assertion'ı dosyanın geçmişini anlatır:
            - c2pa.created    → Oluşturulmuş (AI olabilir)
            - c2pa.generated  → Üretilmiş (genellikle AI)
            - c2pa.captured   → Kamerayla çekilmiş (gerçek)
            - c2pa.edited     → Düzenlenmiş
            - c2pa.drawing    → Çizilmiş

        Returns:
            {
                "actions_found": list[str],
                "has_ai_actions": bool,
                "has_capture_actions": bool,
                "has_edit_actions": bool,
                "action_details": list[dict]
            }
        """
        assertions = manifest.get("assertions", [])
        actions_found = []
        action_details = []
        has_ai = False
        has_capture = False
        has_edit = False

        for assertion in assertions:
            label = assertion.get("label", "")
            data = assertion.get("data", {})

            # c2pa.actions (v1) VEYA c2pa.actions.v2 — ikisini de yakala
            if "c2pa.actions" in label:
                actions = data.get("actions", [])
                for action in actions:
                    action_type = action.get("action", "")
                    actions_found.append(action_type)

                    # softwareAgent: v2=direkt, v1=parameters içinde
                    agent_name = None
                    agent = action.get("softwareAgent")
                    if agent:
                        if isinstance(agent, dict):
                            agent_name = agent.get("name")
                        elif isinstance(agent, str):
                            agent_name = agent
                    else:
                        params = action.get("parameters", {})
                        if isinstance(params, dict):
                            param_agent = params.get("softwareAgent")
                            if isinstance(param_agent, dict):
                                agent_name = param_agent.get("name")
                            elif param_agent:
                                agent_name = str(param_agent)

                    detail = {
                        "action": action_type,
                        "softwareAgent": agent_name,
                        "digitalSourceType": action.get("digitalSourceType"),
                        "description": action.get("parameters", {}).get("description")
                            if isinstance(action.get("parameters"), dict) else None,
                    }
                    action_details.append(detail)

                    if action_type in self.ai_actions:
                        has_ai = True
                    if action_type in self.non_ai_actions:
                        has_capture = True
                    if action_type in self.edit_actions:
                        has_edit = True

        return {
            "actions_found": actions_found,
            "has_ai_actions": has_ai,
            "has_capture_actions": has_capture,
            "has_edit_actions": has_edit,
            "action_details": action_details
        }

    def _analyze_digital_source_type(self, manifest: dict) -> dict:
        """
        IPTC digitalSourceType assertion'ını analiz eder.

        Bu alan doğrudan AI üretimi olup olmadığını belirtir:
            - trainedAlgorithmicMedia → AI tarafından üretilmiş
            - algorithmicMedia → Algoritma ile üretilmiş
            - compositeWithTrainedAlgorithmicMedia → AI ile düzenlenmiş

        NOT: Bir manifest'te birden fazla digitalSourceType olabilir.
        Örneğin Google Gemini:
            c2pa.edited → trainedAlgorithmicMedia
            c2pa.edited → composite
        Bu durumda AI olan kaynak tipi ÖNCELİKLİDİR.

        Returns:
            {
                "source_type": str | None,
                "all_source_types": list[str],
                "is_ai_source": bool,
                "source_category": str
            }
        """
        assertions = manifest.get("assertions", [])
        all_found = []  # Bulunan tüm source type'lar

        for assertion in assertions:
            label = assertion.get("label", "")
            data = assertion.get("data", {})

            # c2pa.digital_source_type veya IPTC assertion
            if "digital_source_type" in label.lower() or "digitalsourcetype" in str(data).lower():
                if isinstance(data, str):
                    all_found.append(data)
                elif isinstance(data, dict):
                    val = data.get("digitalSourceType") or data.get("value")
                    if val:
                        all_found.append(val)

            # c2pa.actions assertion'ında gömülü olabilir
            if "c2pa.actions" in label:
                for action in data.get("actions", []):
                    ds_type = action.get("digitalSourceType")
                    if ds_type:
                        all_found.append(ds_type)

        # EXIF assertion'larında da olabilir
        for assertion in assertions:
            data = assertion.get("data", {})
            if isinstance(data, dict):
                ds = data.get("dc:source") or data.get("digitalSourceType")
                if ds and "iptc.org" in str(ds):
                    all_found.append(ds)

        # Tekrarları temizle (sırayı koru)
        seen = set()
        unique_found = []
        for t in all_found:
            if t not in seen:
                unique_found.append(t)
                seen.add(t)

        # Öncelik: AI source type varsa onu seç (en güçlü kanıt)
        best_type = None
        best_ai = False
        best_category = "none"

        for src in unique_found:
            src_lower = src.lower()
            if "trainedalgorithmicmedia" in src_lower:
                best_type = src
                best_ai = True
                best_category = "ai_generated"
                break  # En güçlü → başka aramaya gerek yok
            elif "compositewithtrainedalgorithmic" in src_lower:
                if best_category not in ("ai_generated",):
                    best_type = src
                    best_ai = True
                    best_category = "ai_edited"
            elif "algorithmicmedia" in src_lower:
                if best_category not in ("ai_generated", "ai_edited"):
                    best_type = src
                    best_ai = True
                    best_category = "algorithmic"
            elif not best_type:
                best_type = src
                best_category = "other"

        return {
            "source_type": best_type,
            "all_source_types": unique_found,
            "is_ai_source": best_ai,
            "source_category": best_category
        }

    def _analyze_ingredients(self, manifest: dict) -> dict:
        """
        Kaynak materyal zincirini analiz eder.

        Ingredients, dosyanın kaynağını gösterir:
            - parentOf → Ana kaynak (orijinal dosya)
            - componentOf → Bileşen
            - inputTo → Girdi olarak kullanılmış

        Returns:
            {
                "has_ingredients": bool,
                "ingredient_count": int,
                "ingredients_summary": list[dict]
            }
        """
        ingredients = manifest.get("ingredients", [])

        if not ingredients:
            return {
                "has_ingredients": False,
                "ingredient_count": 0,
                "ingredients_summary": []
            }

        summaries = []
        for ing in ingredients:
            summary = {
                "title": ing.get("title"),
                "format": ing.get("format"),
                "relationship": ing.get("relationship"),
                "has_manifest": ing.get("manifest") is not None,
            }
            summaries.append(summary)

        return {
            "has_ingredients": True,
            "ingredient_count": len(ingredients),
            "ingredients_summary": summaries
        }

    def _extract_validation(self, manifest_data: dict) -> dict:
        """
        C2PA imza doğrulama durumunu çıkarır.

        c2pa-python Reader otomatik olarak imza doğrulama yapar.
        validation_status alanı varsa sorun var demektir.
        Yoksa → imza geçerli.

        Returns:
            {
                "is_valid": bool,
                "validation_errors": list[str],
                "error_count": int
            }
        """
        val_status = manifest_data.get("validation_status")

        if val_status is None:
            # validation_status yoksa → imza geçerli
            return {
                "is_valid": True,
                "validation_errors": [],
                "error_count": 0
            }

        # validation_status varsa → sorunlar var
        errors = []
        if isinstance(val_status, list):
            for item in val_status:
                if isinstance(item, dict):
                    code = item.get("code", "unknown")
                    explanation = item.get("explanation", "")
                    errors.append(f"{code}: {explanation}")
                else:
                    errors.append(str(item))
        elif isinstance(val_status, dict):
            errors.append(str(val_status))

        return {
            "is_valid": len(errors) == 0,
            "validation_errors": errors,
            "error_count": len(errors)
        }

    # ════════════════════════════════════════════════════════════════
    # MANİFEST ZİNCİR TARAMASI (CHAIN SCANNING)
    # ════════════════════════════════════════════════════════════════

    def _scan_full_chain(self, all_manifests: dict,
                         active_id: str | None) -> dict:
        """
        Tüm manifest zincirini tarar ve AI sinyallerini toplar.

        OpenAI örneği:
            Manifest 1 (parent): c2pa.created, GPT-4o, trainedAlgorithmicMedia
            Manifest 2 (active): c2pa.opened (bilgi yok)

        Bu metod parent manifest'lerdeki bilgiyi de yakalar.

        Returns:
            {
                "chain_length": int,
                "all_issuers": list[str],
                "all_software_agents": list[str],
                "all_actions": list[str],
                "all_digital_source_types": list[str],
                "all_timestamps": list[str],
                "all_generators": list[str],
                "parent_manifests": list[dict]
            }
        """
        all_issuers = []
        all_agents = []
        all_actions = []
        all_source_types = []
        all_timestamps = []
        all_generators = []
        parent_details = []

        for label, manifest in all_manifests.items():
            is_active = (label == active_id)

            # Signature info
            sig_info = manifest.get("signature_info", {})
            issuer = sig_info.get("issuer")
            if issuer:
                all_issuers.append(issuer)
            sig_time = sig_info.get("time")
            if sig_time:
                all_timestamps.append(sig_time)

            # Generator info
            gen_info = manifest.get("claim_generator_info", [])
            if gen_info and isinstance(gen_info, list):
                for info in gen_info:
                    name = info.get("name") if isinstance(info, dict) else None
                    if name:
                        all_generators.append(name)
            claim_gen = manifest.get("claim_generator")
            if claim_gen:
                all_generators.append(claim_gen)

            # Assertions
            for assertion in manifest.get("assertions", []):
                a_label = assertion.get("label", "")
                data = assertion.get("data", {})

                if "c2pa.actions" in a_label:
                    for action in data.get("actions", []):
                        action_type = action.get("action", "")
                        if action_type:
                            all_actions.append(action_type)

                        # softwareAgent (v2 format: dict, v1: params)
                        agent = action.get("softwareAgent")
                        if agent:
                            if isinstance(agent, dict):
                                name = agent.get("name", "")
                                if name:
                                    all_agents.append(name)
                            elif isinstance(agent, str):
                                all_agents.append(agent)
                        else:
                            params = action.get("parameters", {})
                            if isinstance(params, dict):
                                pa = params.get("softwareAgent")
                                if pa:
                                    if isinstance(pa, dict):
                                        all_agents.append(pa.get("name", ""))
                                    else:
                                        all_agents.append(str(pa))

                        # digitalSourceType
                        dst = action.get("digitalSourceType")
                        if dst:
                            all_source_types.append(dst)

            if not is_active:
                parent_details.append({
                    "label": label,
                    "issuer": issuer,
                    "generators": [
                        info.get("name") for info in gen_info
                        if isinstance(info, dict) and info.get("name")
                    ] if gen_info and isinstance(gen_info, list) else []
                })

        return {
            "chain_length": len(all_manifests),
            "all_issuers": list(set(all_issuers)),
            "all_software_agents": list(set(all_agents)),
            "all_actions": list(set(all_actions)),
            "all_digital_source_types": list(set(all_source_types)),
            "all_timestamps": all_timestamps,
            "all_generators": list(set(all_generators)),
            "parent_manifests": parent_details
        }

    def _enrich_from_chain(self, creator_info: dict, tool_info: dict,
                           timestamp_info: dict, action_analysis: dict,
                           source_type_analysis: dict,
                           chain_info: dict) -> tuple:
        """
        Active manifest'te eksik kalan bilgileri zincirden doldurur.

        Kural: Active manifest'teki değer varsa dokunma.
        Sadece eksik (None / empty) alanları zincirden al.
        Zincirden alınan bilgilere "(from_chain)" notu ekle.
        """

        # ── Creator: issuer eksikse zincirden al ──
        if not creator_info.get("issuer"):
            for issuer in chain_info.get("all_issuers", []):
                creator_info["issuer"] = issuer
                creator_info["from_chain"] = True
                # AI issuer kontrolü
                for known, key in self.known_issuers.items():
                    if known.lower() in issuer.lower():
                        creator_info["is_known_ai_issuer"] = True
                        creator_info["matched_issuer_key"] = key
                        break
                break

        # ── Tool: software_agent eksikse zincirden al ──
        # Bilinen AI aracı olanı tercih et (GPT-4o > OpenAI API)
        if not tool_info.get("software_agent"):
            chain_agents = chain_info.get("all_software_agents", [])

            # Önce bilinen AI aracı olanı bul
            best_agent = None
            best_tool_match = None
            for agent in chain_agents:
                for known_agent in self.known_agents:
                    if known_agent.lower() in agent.lower():
                        best_agent = agent
                        best_tool_match = known_agent
                        break
                if best_agent:
                    break

            # Bilinen yoksa ilk agent'ı al
            if not best_agent and chain_agents:
                best_agent = chain_agents[0]

            if best_agent:
                tool_info["software_agent"] = best_agent
                tool_info["from_chain"] = True
                if best_tool_match:
                    tool_info["is_known_ai_tool"] = True
                    tool_info["matched_tool"] = best_tool_match

        # ── Tool: claim_generator_parsed eksikse zincirden al ──
        if not tool_info.get("claim_generator_parsed"):
            for gen in chain_info.get("all_generators", []):
                parsed = gen.split("/")[0].strip() if gen else gen
                tool_info["claim_generator_parsed"] = parsed
                # AI tool kontrolü
                if not tool_info.get("is_known_ai_tool"):
                    for known_agent in self.known_agents:
                        if known_agent.lower() in parsed.lower():
                            tool_info["is_known_ai_tool"] = True
                            tool_info["matched_tool"] = known_agent
                            break
                break

        # ── Timestamp: eksikse zincirden al ──
        if not timestamp_info.get("signature_time"):
            for ts in chain_info.get("all_timestamps", []):
                timestamp_info["signature_time"] = ts
                timestamp_info["has_timestamp"] = True
                timestamp_info["from_chain"] = True
                break

        # ── Actions: zincirdeki tüm eylemleri birleştir ──
        chain_actions = chain_info.get("all_actions", [])
        existing_actions = set(action_analysis.get("actions_found", []))
        for action in chain_actions:
            if action not in existing_actions:
                action_analysis["actions_found"].append(action)
                existing_actions.add(action)
                if action in self.ai_actions:
                    action_analysis["has_ai_actions"] = True
                if action in self.non_ai_actions:
                    action_analysis["has_capture_actions"] = True
                if action in self.edit_actions:
                    action_analysis["has_edit_actions"] = True

        # Zincirden gelen action detail'lerini de ekle
        chain_agents = chain_info.get("all_software_agents", [])
        chain_dsts = chain_info.get("all_digital_source_types", [])
        for action in chain_actions:
            if action not in [d["action"] for d in action_analysis["action_details"]]:
                detail = {
                    "action": action,
                    "softwareAgent": chain_agents[0] if chain_agents else None,
                    "digitalSourceType": chain_dsts[0] if chain_dsts else None,
                    "description": None,
                    "from_chain": True
                }
                action_analysis["action_details"].append(detail)

        # ── Digital Source Type: eksikse VEYA AI değilse zincirden al ──
        # Kurallar:
        #   1. source_type null → zincirden al
        #   2. source_type var ama AI değil + zincirde AI var → AI olanı al
        #   3. source_type AI → dokunma
        current_is_ai = source_type_analysis.get("is_ai_source", False)
        current_empty = not source_type_analysis.get("source_type")

        if current_empty or not current_is_ai:
            for dst in chain_info.get("all_digital_source_types", []):
                dst_lower = dst.lower()
                # Sadece AI source type'ları tercih et
                if "trainedalgorithmicmedia" in dst_lower:
                    source_type_analysis["source_type"] = dst
                    source_type_analysis["is_ai_source"] = True
                    source_type_analysis["source_category"] = "ai_generated"
                    source_type_analysis["from_chain"] = True
                    break
                elif "compositewithtrainedalgorithmic" in dst_lower:
                    source_type_analysis["source_type"] = dst
                    source_type_analysis["is_ai_source"] = True
                    source_type_analysis["source_category"] = "ai_edited"
                    source_type_analysis["from_chain"] = True
                    break
                elif "algorithmicmedia" in dst_lower:
                    source_type_analysis["source_type"] = dst
                    source_type_analysis["is_ai_source"] = True
                    source_type_analysis["source_category"] = "algorithmic"
                    source_type_analysis["from_chain"] = True
                    break

        return (creator_info, tool_info, timestamp_info,
                action_analysis, source_type_analysis)

    # ════════════════════════════════════════════════════════════════
    # SKORLAMA
    # ════════════════════════════════════════════════════════════════

    def _calculate_score(self, creator_info: dict, tool_info: dict,
                         action_analysis: dict, source_type_analysis: dict,
                         ingredient_analysis: dict,
                         validation_info: dict) -> tuple[float, dict]:
        """
        C2PA provenance verilerinden dinamik skor hesaplar.

        C2PA bulunduysa (bu metot sadece C2PA varken çağrılır):

        Baz skor: 0.40 (C2PA var = en azından dijital araç kullanılmış)

        Artırıcı sinyaller (kesin kanıtlar):
            - digitalSourceType = trainedAlgorithmicMedia  → +0.35
            - digitalSourceType = compositeWithTrained...  → +0.25
            - Bilinen AI issuer (OpenAI, Google, vb.)      → +0.15
            - Bilinen AI araç (DALL-E, Gemini, vb.)        → +0.15
            - AI eylemleri (c2pa.created, c2pa.generated)  → +0.10
            - İmza geçerli (doğrulama başarılı)            → +0.05

        Azaltıcı sinyaller (gerçek fotoğraf kanıtları):
            - Kamera eylemi (c2pa.captured)                → -0.30
            - İmza doğrulama hatalı                        → -0.10

        Skor: max(0.0, min(1.0, toplam))
        """
        breakdown = {}
        score = 0.40  # C2PA var → baz skor
        breakdown["base_c2pa_found"] = {
            "signal": "C2PA manifest mevcut",
            "contribution": 0.40
        }

        # ── Artırıcı: digitalSourceType ──
        if source_type_analysis["is_ai_source"]:
            category = source_type_analysis["source_category"]
            if category == "ai_generated":
                bonus = 0.35
            elif category == "ai_edited":
                bonus = 0.25
            elif category == "algorithmic":
                bonus = 0.20
            else:
                bonus = 0.10
            score += bonus
            breakdown["digital_source_type"] = {
                "signal": f"{category} ({source_type_analysis['source_type']})",
                "contribution": bonus
            }

        # ── Artırıcı: Bilinen AI issuer ──
        if creator_info["is_known_ai_issuer"]:
            bonus = 0.15
            score += bonus
            breakdown["known_ai_issuer"] = {
                "signal": f"{creator_info['issuer']} → {creator_info['matched_issuer_key']}",
                "contribution": bonus
            }

        # ── Artırıcı: Bilinen AI araç ──
        if tool_info["is_known_ai_tool"]:
            bonus = 0.15
            score += bonus
            breakdown["known_ai_tool"] = {
                "signal": f"{tool_info['matched_tool']} (claim_generator: {tool_info['claim_generator_parsed']})",
                "contribution": bonus
            }

        # ── Artırıcı: AI eylemleri ──
        if action_analysis["has_ai_actions"]:
            bonus = 0.10
            score += bonus
            breakdown["ai_actions"] = {
                "signal": f"AI eylemleri tespit edildi: {action_analysis['actions_found']}",
                "contribution": bonus
            }

        # ── Artırıcı: İmza geçerli ──
        if validation_info["is_valid"]:
            bonus = 0.05
            score += bonus
            breakdown["signature_valid"] = {
                "signal": "C2PA imzasi gecerli (dogrulama basarili)",
                "contribution": bonus
            }

        # ── Azaltıcı: Kamera eylemi ──
        if action_analysis["has_capture_actions"]:
            penalty = -0.30
            score += penalty
            breakdown["capture_action"] = {
                "signal": "Kamera/tarama eylemi tespit edildi (gercek icerik)",
                "contribution": penalty
            }

        # ── Azaltıcı: İmza doğrulama hatası ──
        if not validation_info["is_valid"]:
            penalty = -0.10
            score += penalty
            breakdown["validation_error"] = {
                "signal": f"Imza dogrulama hatasi ({validation_info['error_count']} hata)",
                "contribution": penalty
            }

        # Sınırla
        final_score = round(max(0.0, min(1.0, score)), 4)

        breakdown["_total"] = {
            "raw_sum": round(score, 4),
            "final_score": final_score
        }

        return final_score, breakdown

    def _determine_verdict(self, score: float) -> str:
        """Skora göre verdict belirler."""
        if score >= 0.70:
            return "high_risk"
        elif score >= 0.40:
            return "medium_risk"
        else:
            return "low_risk"

    # ════════════════════════════════════════════════════════════════
    # RAPORLAMA
    # ════════════════════════════════════════════════════════════════

    def _generate_details(self, score: float, verdict: str,
                          creator_info: dict, tool_info: dict,
                          timestamp_info: dict, action_analysis: dict,
                          source_type_analysis: dict,
                          validation_info: dict) -> str:
        """Analiz sonucunun Türkçe açıklamasını üretir."""
        parts = []

        parts.append("C2PA CONTENT CREDENTIALS TESPIT EDILDI.")

        # İmzalayan
        issuer = creator_info.get("issuer", "Bilinmiyor")
        parts.append(f"Imzalayan: {issuer}")
        if creator_info["is_known_ai_issuer"]:
            parts.append(
                f"  → Bilinen AI uretim kaynagi: {creator_info['matched_issuer_key']}"
            )

        # Araç
        tool = tool_info.get("claim_generator_parsed") or tool_info.get("claim_generator") or "Bilinmiyor"
        sw_agent = tool_info.get("software_agent")
        parts.append(f"Ureten arac: {tool}")
        if sw_agent:
            parts.append(f"  → AI modeli: {sw_agent}")
        if tool_info["is_known_ai_tool"]:
            parts.append(
                f"  → Bilinen AI araci: {tool_info['matched_tool']}"
            )

        # Tarih
        if timestamp_info["has_timestamp"]:
            parts.append(f"Imza tarihi: {timestamp_info['signature_time']}")
        else:
            parts.append("Imza tarihi: Mevcut degil")

        # Dijital kaynak tipi
        if source_type_analysis["is_ai_source"]:
            parts.append(
                f"DIJITAL KAYNAK TIPI: {source_type_analysis['source_category'].upper()} "
                f"({source_type_analysis['source_type']}). "
                f"Bu alan dosyanin AI tarafindan uretildigini dogrudan belirtmektedir."
            )

        # Eylemler
        if action_analysis["actions_found"]:
            actions_str = ", ".join(action_analysis["actions_found"])
            parts.append(f"C2PA eylemleri: [{actions_str}]")
            if action_analysis["has_capture_actions"]:
                parts.append(
                    "  → Kamera/tarama eylemi tespit edildi — gercek icerik sinyali."
                )

        # İmza doğrulama
        if validation_info["is_valid"]:
            parts.append("Imza dogrulama: GECERLI (kriptografik dogrulama basarili)")
        else:
            err_count = validation_info["error_count"]
            parts.append(
                f"Imza dogrulama: HATALI ({err_count} hata tespit edildi)"
            )

        return " | ".join(parts)

    def _build_no_c2pa_results(self) -> dict:
        """C2PA bulunamadığında dönen standart sonuç yapısı."""
        return {
            "has_c2pa": False,
            "creator": {
                "issuer": None,
                "is_known_ai_issuer": False,
                "matched_issuer_key": None
            },
            "tool": {
                "claim_generator": None,
                "claim_generator_parsed": None,
                "software_agent": None,
                "is_known_ai_tool": False,
                "matched_tool": None
            },
            "timestamp": {
                "signature_time": None,
                "has_timestamp": False
            },
            "actions": {
                "actions_found": [],
                "has_ai_actions": False,
                "has_capture_actions": False,
                "has_edit_actions": False,
                "action_details": []
            },
            "digital_source_type": {
                "source_type": None,
                "is_ai_source": False,
                "source_category": "none"
            },
            "ingredients": {
                "has_ingredients": False,
                "ingredient_count": 0,
                "ingredients_summary": []
            },
            "validation": {
                "is_valid": False,
                "validation_errors": [],
                "error_count": 0
            },
            "score_breakdown": {},
            "manifest_count": 0,
            "active_manifest_id": None
        }