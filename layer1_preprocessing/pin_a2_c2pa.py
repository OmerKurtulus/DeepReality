"""
DeepReality — PIN-A2: C2PA Provenance Analysis
==============================================

Purpose:
    Determine whether the file carries C2PA Content Credentials and, if
    so, extract in structured form who produced it, with which tool, at
    what time, and whether the signature validates.

    This is the most authoritative instrument in the system. Every
    other pin infers provenance from appearance; this one reads a
    cryptographically signed assertion made by the producing tool.

Difference from PIN-A1:
    PIN-A1 -> heuristic byte-pattern scan of the container
    PIN-A2 -> manifest parsing and validation via the reference
              c2pa-python implementation

Output:
    has_c2pa        : bool  — was a C2PA manifest found?
    creator         : str   — signing organisation (signature_info.issuer)
    tool            : str   — producing tool (claim_generator / softwareAgent)
    timestamp       : str   — signature time (ISO 8601)
    is_ai_generated : bool  — is there evidence of AI generation?
    ai_source_type  : str   — IPTC digital source type
    actions         : list  — C2PA action history
    validation      : dict  — signature validation state
    ingredients     : list  — chain of source material
    score           : float — 0.0 (no C2PA / authentic) to 1.0 (certain AI)

Library:
    c2pa-python >= 0.28.0 (pip install c2pa-python)
    https://github.com/contentauth/c2pa-python

Author: Omer Faruk Kurtulus
"""

import json
import traceback
from pathlib import Path

from core.base_pin import BasePin
from config.settings import C2PA_CONFIG


class PinA2C2pa(BasePin):
    """
    PIN-A2: C2PA Content Credentials provenance analysis.

    Reads, validates and structurally decomposes the C2PA manifest of a
    file using the c2pa-python Reader.
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
    # MAIN ANALYSIS
    # ════════════════════════════════════════════════════════════════

    def analyze(self, file_path: str) -> dict:
        """
        C2PA manifest reading and analysis pipeline.

        Steps:
            1. Open the file with c2pa.Reader
            2. Locate the active manifest
            3. Scan the ENTIRE manifest chain, parents included
            4. claim_generator     -> producing tool
            5. signature_info      -> signer and signature time
            6. assertions          -> actions and digital source type
            7. ingredients         -> source material chain
            8. validation_status   -> signature validation
            9. Score every collected signal

        Note: some providers, OpenAI among them, emit a two-manifest
        chain in which the informative claim sits in the parent:
            Manifest 1 (parent): c2pa.created + GPT-4o +
                                 trainedAlgorithmicMedia
            Manifest 2 (active): c2pa.opened — merely "file was opened"
        Scanning only the active manifest would therefore miss the
        decisive evidence entirely, which is why the full chain is
        traversed.
        """

        # ── Step 1: read the C2PA manifest ──
        manifest_data = self._read_c2pa_manifest(file_path)

        if not manifest_data["has_c2pa"]:
            # No C2PA -> this pin has no data to contribute, score 0
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

        # ── Step 2: extract data from the active manifest ──
        active_manifest = manifest_data["active_manifest"]

        creator_info = self._extract_creator(active_manifest)
        tool_info = self._extract_tool(active_manifest)
        timestamp_info = self._extract_timestamp(active_manifest)
        action_analysis = self._analyze_actions(active_manifest)
        source_type_analysis = self._analyze_digital_source_type(active_manifest)
        ingredient_analysis = self._analyze_ingredients(active_manifest)
        validation_info = self._extract_validation(manifest_data)

        # ── Step 3: scan the whole manifest chain (chain enrichment) ──
        # Fields missing from the active manifest are recovered from parents
        all_manifests = manifest_data.get("all_manifests", {})
        active_id = manifest_data.get("active_manifest_id")
        chain_info = self._scan_full_chain(all_manifests, active_id)

        # Enrichment: fill the gaps from the chain
        creator_info, tool_info, timestamp_info, action_analysis, \
            source_type_analysis = self._enrich_from_chain(
                creator_info, tool_info, timestamp_info,
                action_analysis, source_type_analysis, chain_info
            )

        # ── Step 4: scoring ──
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
                    "read_error": None  # Not an error: the file simply carries no C2PA data
                }
            else:
                # A genuine failure
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
    # EXTRACTION
    # ════════════════════════════════════════════════════════════════

    def _extract_creator(self, manifest: dict) -> dict:
        """
        Extract the signing organisation.

        Sources:
            - signature_info.issuer   (primary)
            - claim_generator_info    (secondary)

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
        Extract the producing tool.

        Sources:
            - claim_generator (primary, e.g. "DALL-E 3/1.0 c2pa-rs/0.33.0")
            - claim_generator_info (secondary)
            - softwareAgent within the assertions (tertiary)

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

        # claim_generator usually follows "ToolName/version sdk/version"
        claim_gen_parsed = None
        if claim_gen:
            # Take the first component, which names the tool
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

        # Is this a known generative tool?
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
        Search the assertions for a softwareAgent field.

        Two encodings are supported:
        - c2pa.actions v1: action.parameters.softwareAgent
        - c2pa.actions v2: action.softwareAgent (dict or string)
        """
        assertions = manifest.get("assertions", [])
        for assertion in assertions:
            label = assertion.get("label", "")
            data = assertion.get("data", {})

            if "c2pa.actions" in label:
                actions = data.get("actions", [])
                for action in actions:

                    # v2 encoding: softwareAgent sits directly on the action (dict)
                    agent = action.get("softwareAgent")
                    if agent:
                        if isinstance(agent, dict):
                            name = agent.get("name", "")
                            if name:
                                return name
                        elif isinstance(agent, str):
                            return agent

                    # v1 encoding: softwareAgent sits under parameters
                    params = action.get("parameters", {})
                    if isinstance(params, dict):
                        param_agent = params.get("softwareAgent")
                        if param_agent:
                            if isinstance(param_agent, dict):
                                return param_agent.get("name", str(param_agent))
                            return str(param_agent)

            # It may also appear inside stds.schema-org.CreativeWork
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
        Extract the signature timestamp.

        Sources:
            - signature_info.time         (primary)
            - claim_generator_info[].time (secondary)

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
        Analyse the actions recorded in the C2PA assertions.

        The c2pa.actions assertion narrates the history of the file:
            - c2pa.created    -> created (possibly by AI)
            - c2pa.generated  -> generated (usually by AI)
            - c2pa.captured   -> captured with a camera (authentic)
            - c2pa.edited     -> edited
            - c2pa.drawing    -> drawn

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

                    # softwareAgent: v2 places it directly, v1 under parameters
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
        Analyse the IPTC digitalSourceType assertion.

        This field states directly whether the asset was synthesised:
            - trainedAlgorithmicMedia              -> produced by AI
            - algorithmicMedia                     -> algorithmically produced
            - compositeWithTrainedAlgorithmicMedia -> AI-assisted edit

        Note: a manifest may declare several digital source types. Google
        Gemini, for example, emits:
            c2pa.edited -> trainedAlgorithmicMedia
            c2pa.edited -> composite
        The AI-bearing type TAKES PRECEDENCE in that situation, since it
        is the stronger and more specific claim.

        Returns:
            {
                "source_type": str | None,
                "all_source_types": list[str],
                "is_ai_source": bool,
                "source_category": str
            }
        """
        assertions = manifest.get("assertions", [])
        all_found = []  # Every source type encountered

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

            # It may be embedded in the c2pa.actions assertion
            if "c2pa.actions" in label:
                for action in data.get("actions", []):
                    ds_type = action.get("digitalSourceType")
                    if ds_type:
                        all_found.append(ds_type)

        # It may also appear in EXIF assertions
        for assertion in assertions:
            data = assertion.get("data", {})
            if isinstance(data, dict):
                ds = data.get("dc:source") or data.get("digitalSourceType")
                if ds and "iptc.org" in str(ds):
                    all_found.append(ds)

        # Remove duplicates while preserving order
        seen = set()
        unique_found = []
        for t in all_found:
            if t not in seen:
                unique_found.append(t)
                seen.add(t)

        # Precedence: prefer an AI source type, the strongest evidence
        best_type = None
        best_ai = False
        best_category = "none"

        for src in unique_found:
            src_lower = src.lower()
            if "trainedalgorithmicmedia" in src_lower:
                best_type = src
                best_ai = True
                best_category = "ai_generated"
                break  # Strongest available — no need to keep searching
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
        Analyse the chain of source material.

        Ingredients describe where the asset came from:
            - parentOf    -> principal source (the original file)
            - componentOf -> a component of the composition
            - inputTo     -> consumed as an input

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
        Extract the signature validation state.

        The c2pa-python Reader validates signatures automatically. The
        presence of a validation_status field indicates a problem; its
        absence means the signature validated cleanly.

        Returns:
            {
                "is_valid": bool,
                "validation_errors": list[str],
                "error_count": int
            }
        """
        val_status = manifest_data.get("validation_status")

        if val_status is None:
            # No validation_status -> the signature is valid
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
    # MANIFEST CHAIN SCANNING
    # ════════════════════════════════════════════════════════════════

    def _scan_full_chain(self, all_manifests: dict,
                         active_id: str | None) -> dict:
        """
        Traverse the entire manifest chain and collect AI signals.

        Illustrative OpenAI case:
            Manifest 1 (parent): c2pa.created, GPT-4o,
                                 trainedAlgorithmicMedia
            Manifest 2 (active): c2pa.opened, carrying no information

        This method recovers the evidence held in parent manifests.

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
        Fill gaps in the active manifest from the rest of the chain.

        Rule: never overwrite a value the active manifest already
        provides. Only fields that are absent (None or empty) are taken
        from the chain, and every value sourced that way is annotated
        with "(from_chain)" so the provenance of the provenance data
        itself remains auditable.
        """

        # ── Creator: issuer eksikse zincirden al ──
        if not creator_info.get("issuer"):
            for issuer in chain_info.get("all_issuers", []):
                creator_info["issuer"] = issuer
                creator_info["from_chain"] = True
                # AI issuer check
                for known, key in self.known_issuers.items():
                    if known.lower() in issuer.lower():
                        creator_info["is_known_ai_issuer"] = True
                        creator_info["matched_issuer_key"] = key
                        break
                break

        # ── Tool: software_agent eksikse zincirden al ──
        # Prefer the more specific known tool (GPT-4o over OpenAI API)
        if not tool_info.get("software_agent"):
            chain_agents = chain_info.get("all_software_agents", [])

            # First look for a recognised generative tool
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

            # Otherwise fall back to the first agent found
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
                # AI tool check
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

        # ── Actions: merge every action in the chain ──
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

        # ── Digital source type: take from the chain if absent or non-AI ──
        # Kurallar:
        #   1. source_type null → zincirden al
        #   2. a source type exists but is not AI, while the chain has one
        #   3. source_type AI → dokunma
        current_is_ai = source_type_analysis.get("is_ai_source", False)
        current_empty = not source_type_analysis.get("source_type")

        if current_empty or not current_is_ai:
            for dst in chain_info.get("all_digital_source_types", []):
                dst_lower = dst.lower()
                # Prefer AI source types only
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
        Compute a graduated score from the C2PA provenance data.

        Invoked only when a manifest was found.

        Base score: 0.40 — the presence of C2PA establishes at minimum
        that a digital tool was involved.

        Aggravating signals (positive evidence of synthesis):
            - digitalSourceType = trainedAlgorithmicMedia  -> +0.35
            - digitalSourceType = compositeWithTrained...   -> +0.25
            - known AI issuer (OpenAI, Google, ...)         -> +0.15
            - known AI tool (DALL-E, Gemini, ...)           -> +0.15
            - generative actions (c2pa.created/generated)   -> +0.10
            - signature validates                           -> +0.05

        Mitigating signals (evidence of authentic capture):
            - camera capture action (c2pa.captured)         -> -0.30
            - signature fails validation                    -> -0.10

        Score: max(0.0, min(1.0, total))
        """
        breakdown = {}
        score = 0.40  # C2PA var → baz skor
        breakdown["base_c2pa_found"] = {
            "signal": "C2PA manifest mevcut",
            "contribution": 0.40
        }

        # ── Aggravating: digitalSourceType ──
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

        # ── Aggravating: known AI issuer ──
        if creator_info["is_known_ai_issuer"]:
            bonus = 0.15
            score += bonus
            breakdown["known_ai_issuer"] = {
                "signal": f"{creator_info['issuer']} → {creator_info['matched_issuer_key']}",
                "contribution": bonus
            }

        # ── Aggravating: known AI tool ──
        if tool_info["is_known_ai_tool"]:
            bonus = 0.15
            score += bonus
            breakdown["known_ai_tool"] = {
                "signal": f"{tool_info['matched_tool']} (claim_generator: {tool_info['claim_generator_parsed']})",
                "contribution": bonus
            }

        # ── Aggravating: generative actions ──
        if action_analysis["has_ai_actions"]:
            bonus = 0.10
            score += bonus
            breakdown["ai_actions"] = {
                "signal": f"AI eylemleri tespit edildi: {action_analysis['actions_found']}",
                "contribution": bonus
            }

        # ── Aggravating: signature validates ──
        if validation_info["is_valid"]:
            bonus = 0.05
            score += bonus
            breakdown["signature_valid"] = {
                "signal": "C2PA imzasi gecerli (dogrulama basarili)",
                "contribution": bonus
            }

        # ── Mitigating: camera capture action ──
        if action_analysis["has_capture_actions"]:
            penalty = -0.30
            score += penalty
            breakdown["capture_action"] = {
                "signal": "Kamera/tarama eylemi tespit edildi (gercek icerik)",
                "contribution": penalty
            }

        # ── Mitigating: signature validation failure ──
        if not validation_info["is_valid"]:
            penalty = -0.10
            score += penalty
            breakdown["validation_error"] = {
                "signal": f"Imza dogrulama hatasi ({validation_info['error_count']} hata)",
                "contribution": penalty
            }

        # Clamp to range
        final_score = round(max(0.0, min(1.0, score)), 4)

        breakdown["_total"] = {
            "raw_sum": round(score, 4),
            "final_score": final_score
        }

        return final_score, breakdown

    def _determine_verdict(self, score: float) -> str:
        """Map the numeric score onto a verdict band."""
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
        """Produce the natural-language explanation of the analysis."""
        parts = []

        parts.append("C2PA CONTENT CREDENTIALS TESPIT EDILDI.")

        # Signer
        issuer = creator_info.get("issuer", "Bilinmiyor")
        parts.append(f"Imzalayan: {issuer}")
        if creator_info["is_known_ai_issuer"]:
            parts.append(
                f"  → Bilinen AI uretim kaynagi: {creator_info['matched_issuer_key']}"
            )

        # Tool
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

        # Signature validation
        if validation_info["is_valid"]:
            parts.append("Imza dogrulama: GECERLI (kriptografik dogrulama basarili)")
        else:
            err_count = validation_info["error_count"]
            parts.append(
                f"Imza dogrulama: HATALI ({err_count} hata tespit edildi)"
            )

        return " | ".join(parts)

    def _build_no_c2pa_results(self) -> dict:
        """Standard result returned when no C2PA manifest is present."""
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