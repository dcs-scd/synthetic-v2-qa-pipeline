"""Tests for the NetLogo source parser."""
import pytest
from qa_system_v2bis.synthetic_v2.nlogo_parser import (
    split_nlogo_sections,
    strip_netlogo_comments,
    extract_breeds,
    extract_declared_vars,
    extract_procedure_headers,
    extract_string_literals,
    extract_core_from_code,
)

SAMPLE_CODE = """
breed [agents an-agent]
breed [cops cop]

globals [k threshold]
patches-own [neighborhood]

agents-own [
  risk-aversion
  perceived-hardship
  active?
  jail-term
]

to setup
  clear-all
  ask patches [ set neighborhood (count turtles-here) ]
  create-agents 100 [ set risk-aversion random-float 1.0 ]
  reset-ticks
end

to go
  ask agents [ determine-behavior ]
  tick
end

to determine-behavior  ; agent procedure
  let g grievance
  if g > threshold [ set active? true ]
end

to-report grievance
  report perceived-hardship * (1 - government-legitimacy)
end
"""

SAMPLE_NLOGO = SAMPLE_CODE + "@#$#@#$#@\nSLIDER\n10\n10\ngovernment-legitimacy\ngovernment-legitimacy\n0\n1\n0.8\n0.01\n1\nNIL\nHORIZONTAL\n@#$#@#$#@\nThis is the Info tab.\n\nThe rebellion model simulates citizen uprising against a central authority using Epstein's model of civil violence.\n"


class TestSplitSections:
    def test_splits_three_sections(self):
        sections = split_nlogo_sections(SAMPLE_NLOGO)
        assert "code" in sections
        assert "interface" in sections
        assert "info" in sections
        assert "breed" in sections["code"]
        assert "SLIDER" in sections["interface"]
        assert "rebellion" in sections["info"]


class TestStripComments:
    def test_removes_line_comments(self):
        result = strip_netlogo_comments('let x 5 ; this is a comment\nlet y 10')
        assert "; this is a comment" not in result
        assert "let x 5" in result
        assert "let y 10" in result

    def test_preserves_strings(self):
        result = strip_netlogo_comments('set label "hello ; world"')
        assert 'hello ; world' in result


class TestExtractBreeds:
    def test_extracts_breeds(self):
        code = strip_netlogo_comments(SAMPLE_CODE)
        breeds = extract_breeds(code)
        names = [(b["plural"], b["singular"]) for b in breeds]
        assert ("agents", "an-agent") in names
        assert ("cops", "cop") in names

    def test_dedup_breeds(self):
        code = "breed [cats cat]\nbreed [cats cat]"
        breeds = extract_breeds(code)
        assert len(breeds) == 1


class TestExtractDeclaredVars:
    def test_extracts_globals(self):
        code = strip_netlogo_comments(SAMPLE_CODE)
        decl = extract_declared_vars(code)
        assert "k" in decl["globals"]
        assert "threshold" in decl["globals"]

    def test_extracts_patches_own(self):
        code = strip_netlogo_comments(SAMPLE_CODE)
        decl = extract_declared_vars(code)
        assert "neighborhood" in decl["patches_own"]

    def test_extracts_breed_own(self):
        code = strip_netlogo_comments(SAMPLE_CODE)
        decl = extract_declared_vars(code)
        assert "agents" in decl["breed_own"]
        assert "risk-aversion" in decl["breed_own"]["agents"]
        assert "active?" in decl["breed_own"]["agents"]


class TestExtractProcedures:
    def test_extracts_commands_and_reporters(self):
        code = strip_netlogo_comments(SAMPLE_CODE)
        proc = extract_procedure_headers(code)
        assert "setup" in proc["procedures"]
        assert "go" in proc["procedures"]
        assert "determine-behavior" in proc["procedures"]
        assert "grievance" in proc["procedures"]
        assert proc["procedure_kinds"]["grievance"] == "reporter"
        assert proc["procedure_kinds"]["setup"] == "command"


class TestExtractCoreFromCode:
    def test_complete_extraction(self):
        core = extract_core_from_code(SAMPLE_CODE)
        assert "setup" in core["procedures"]
        assert "grievance" in core["procedures"]
        assert "agents" in core["breeds"]
        assert "cops" in core["breeds"]
        assert "k" in core["variables"]
        assert "risk-aversion" in core["variables"]
        assert "active?" in core["variables"]
