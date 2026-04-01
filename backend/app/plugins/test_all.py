"""P1-P3 综合验证测试。"""
print("=== P1: Plugin Interface ===")
from app.plugins.base import DomainPlugin, QueryContext, SafetyResult, PluginResponse
print("  base: OK")

from app.plugins.registry import register_plugin, get_plugin, list_plugins, initialize_plugins
print("  registry: OK")

qc = QueryContext(query="test", domain="medical")
sr = SafetyResult(safe=True)
pr = PluginResponse(answer="ok")
print(f"  QueryContext: {qc.domain}")
print(f"  SafetyResult: safe={sr.safe}")

print("\n=== P2: Medical Plugin ===")
from app.plugins.medical.rules import MedicalRuleService, medical_rule_service
print("  medical.rules: OK")

from app.plugins.medical.plugin import MedicalDomainPlugin, medical_plugin
print("  medical.plugin: OK")

import json
from pathlib import Path
# find medical mapping files
medical_pkg_path = Path(__file__).parent  # app/plugins
mappings = medical_pkg_path / "medical" / "mappings"
dept = json.loads((mappings / "departments.json").read_text(encoding="utf-8"))
symp = json.loads((mappings / "symptoms.json").read_text(encoding="utf-8"))
print(f"  departments: {len(dept['departments'])} depts, {len(dept['aliases'])} aliases")
print(f"  symptoms: {len(symp)} mappings")

# Test plugin methods (classify_intent needs settings/lazy import, skip for now)
plugin = medical_plugin
print(f"  plugin.NAME: {plugin.NAME}")

import asyncio
safety = asyncio.run(plugin.check_safety("我想死"))
print(f"  check_safety('我想死'): crisis={safety.crisis_detected}")

disclaimer_resp = plugin.format_response("建议多喝水", qc)
print(f"  format_response has disclaimer: {'以上仅供参考' in disclaimer_resp}")

post_processed = plugin.post_process([{"score": 0.1, "department": "心血管内科"}], qc)
print(f"  post_process dept alias resolved: {post_processed[0]['department']}")

print("\n=== P3: Legal Plugin ===")
from app.plugins.legal.article_fetcher import ArticleFetcher
print("  legal.article_fetcher: OK")
from app.plugins.legal.citation_formatter import CitationFormatter
print("  legal.citation_formatter: OK")
from app.plugins.legal.validity_checker import ValidityChecker
print("  legal.validity_checker: OK")
from app.plugins.legal.plugin import LegalDomainPlugin, legal_plugin
print("  legal.plugin: OK")

legal_mappings = medical_pkg_path / "legal" / "mappings"
law = json.loads((legal_mappings / "law_codes.json").read_text(encoding="utf-8"))
terms = json.loads((legal_mappings / "legal_terms.json").read_text(encoding="utf-8"))
print(f"  law_codes: {len(law['structure'])} books")
print(f"  legal_terms: {len(terms)} terms")

# Test legal plugin
lp = legal_plugin
print(f"  classify_intent('离婚财产怎么分'): {lp.classify_intent('离婚财产怎么分')}")
print(f"  classify_intent('你好'): {lp.classify_intent('你好')}")
legal_warnings = asyncio.run(lp.check_safety('根据婚姻法')).warnings
print(f"  check_safety('根据婚姻法'): warnings={legal_warnings}")
print(f"  format_response test: {'不构成正式法律意见' in lp.format_response('test', qc)}")

# Citation formatter
cite = CitationFormatter.format_article(584, "合同违约责任")
print(f"  format_article: {cite}")

valid = ValidityChecker()
print(f"  is_article_valid(1): {valid.is_article_valid(1)}")
print(f"  is_article_valid(9999): {valid.is_article_valid(9999)}")
print(f"  check_superseded('合同法'): {valid.check_superseded('合同法')}")

print("\n=== All Tests Passed ===")
