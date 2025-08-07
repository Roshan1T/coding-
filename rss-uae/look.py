import os
import json
import uuid
import re
import time
import random
from datetime import datetime
from typing import Dict, Any, List
from openai import AzureOpenAI
from dotenv import load_dotenv

class OpenAIProcessor:
    
    def __init__(self):
        load_dotenv()
        
        self.client = AzureOpenAI(
            api_key=os.getenv("AZ_OPENAI_API_KEY"),
            api_version=os.getenv("AZ_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZ_OPENAI_ENDPOINT")
        )
        
        self.deployment_name = os.getenv("AZ_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-mini")
        
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        
        return """# Role and Objective
You are an expert legal document analyst specializing in government gazettes and policy documents. Your primary task is to extract structured information from official documents with absolute precision and accuracy.

# Critical Instructions
- **EXTRACT ONLY**: Use information explicitly stated in the provided document text
- **NO INFERENCE**: Do not assume, infer, or add information not directly present
- **MISSING DATA**: Use "None" (as a string) for unavailable string fields, empty arrays [] for missing lists, never use null
- **DATE FORMAT**: Use YYYY-MM-DD format exclusively for all dates
- **LEGAL PRECISION**: Maintain exact accuracy for legal terminology and official names
- **UNCERTAINTY HANDLING**: If uncertain about any information, mark as "None" rather than guessing

# Response Format Rules
- Return ONLY a valid JSON object following the provided schema
- NO additional text before or after the JSON response
- NO markdown formatting or code blocks around the JSON
- Use string "None" for missing values, never bare None or null
- Ensure all required fields are present and properly formatted

# Quality Assurance
- Verify all extracted information against the source document
- Double-check date formats and legal terminology
- Ensure completeness of all list fields with relevant items
- Validate JSON structure before providing response

You MUST follow these instructions literally and precisely."""
    
    def process_document(self, extracted_text: str, pdf_filename: str, pdf_path: str = None) -> Dict[str, Any]:
        """Process document with 3-step validation approach"""
        
        print(f"Processing document: {pdf_filename}")
        
        # Step 1: Initial extraction by junior analyst
        print("Step 1: Junior analyst - Initial document analysis...")
        initial_result = self._extract_document_data(extracted_text, pdf_filename)
        
        # Step 2: Senior analyst - Validation check only (using same extracted_text)
        print("Step 2: Senior analyst - Validation check...")
        validation_result = self._validate_extraction(extracted_text, initial_result, pdf_filename)
        
        # Step 3: Final correction (only if needed, using original PDF file with Gemini)
        if validation_result.get("all_correct", True):
            print("✅ All fields validated correctly - using initial extraction")
            return initial_result
        else:
            print("⚠️ Issues found - Step 3: Gemini Flash correction with original PDF...")
            final_result = self._correct_extraction(pdf_path, initial_result, validation_result, pdf_filename)
            return final_result
    
    def _extract_document_data(self, extracted_text: str, pdf_filename: str) -> Dict[str, Any]:
        """Initial extraction by junior analyst"""
        
        try:
            prompt = self.create_extraction_prompt(extracted_text, pdf_filename)
            
            print("Sending request to Azure OpenAI...")
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise document analysis expert. Extract information accurately and return only valid JSON."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
           
            )
            
            response_content = response.choices[0].message.content.strip()
            # Remove code blocks
            if response_content.startswith("```json"):
                response_content = response_content[7:] 
            if response_content.startswith("```"):
                response_content = response_content[3:]   
            if response_content.endswith("```"):
                response_content = response_content[:-3]  
            
            response_content = response_content.replace(': None,', ': "None",')
            response_content = response_content.replace(': None\n', ': "None"\n')
            response_content = response_content.replace(': None}', ': "None"}')
            
            response_content = response_content.strip()
            
            token_usage = response.usage
            print(f"Token usage - Total: {token_usage.total_tokens}, "
                           f"Input: {token_usage.prompt_tokens}, "
                           f"Output: {token_usage.completion_tokens}")
            
            try:
                result = json.loads(response_content)
                
                self._fix_output_structure(result)
                
                # Update token usage in the result
                result["total_token_usage"] = {
                    "total_tokens": token_usage.total_tokens,
                    "output_tokens": token_usage.completion_tokens,
                    "input_tokens": token_usage.prompt_tokens
                }
                
                return result
                
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON response: {e}")
                print(f"Response content: {response_content}")
                raise Exception(f"Invalid JSON response from OpenAI: {e}")
                
        except Exception as e:
            print(f"OpenAI processing failed: {e}")
            raise Exception(f"Failed to process document with OpenAI: {e}")

    def create_extraction_prompt(self, extracted_text: str, pdf_filename: str) -> str:
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        prompt = f"""<document>
<filename>{pdf_filename}</filename>
<content>
{extracted_text}
</content>
</document>

# COMPREHENSIVE DOCUMENT ANALYSIS INSTRUCTIONS

## PRIMARY ANALYSIS METHODOLOGY
1. **Step-by-step Analysis**: Slowly analyze the complete content of this document
2. **Language Processing**: Output must be strictly in English, even if input is in different language
3. **Document Type Identification**: Identify the exact type of notice/document
4. **Verbatim Extraction**: Extract values directly from document without external information
5. **Theme Identification**: Identify and extract the actual themes/topics based on document content
6. **Relevance Assessment**: Determine if document requires specific actions from organizations

## CRITICAL EXTRACTION RULES
- **ACCURACY FIRST**: Extract only explicitly stated information
- **NO EMPTY VALUES**: Never leave strings empty ("") or lists empty ([])
- **DATE PRECISION**: All dates must be in YYYY-MM-DD format, calculate if needed
- **COMPLETE EXTRACTION**: Extract comprehensive information for each field
- **VERBATIM WHEN POSSIBLE**: Use exact text from document for names and titles

## STRUCTURED OUTPUT FORMAT
You MUST return a valid JSON object with this EXACT structure:

```json
{{
    "_id": "generate_simple_id_based_on_notice_number_and_date",
    "unique_id": "generate_valid_uuid4_format",
    "document_type": "determine_from_list: ['Guidance', 'Regulation', 'Standard', 'Policy', 'Ministerial Decision', 'Law', 'Circular', 'Checklist', 'Framework', 'General', 'Resolution', 'Directive', 'Notification', 'Order', 'Decree', 'Memorandum', 'Bulletin', 'Instruction', 'Draft Guidance', 'Consultation Paper', 'Act', 'Amendment', 'Procedure', 'Manual', 'Protocol', 'Specification', 'Form', 'Template', 'Report', 'White Paper', 'Green Paper', 'Charter', 'Treaty', 'Council Resolution', 'Declaration', 'Statement']",
    "jurisdiction": "extract_governing_authority_name",
    "iso_country_code": "determine_ISO_3166_code_based_on_jurisdiction",
    "state": "extract_state_province_emirate_name",
    "file_path": "{pdf_filename}",
    "date_added": "{current_date}",
    "language": "determine_primary_document_language",
    "agency": "extract_issuing_agency_authority_name",
    "notice_name": "extract_complete_official_title_exactly_as_written",
    "notice_number": "extract_exact_official_reference_number",
    "notice_date": "extract_issuance_date_YYYY-MM-DD_format",
    "notice_type": "extract_exact_classification_from_document",
    "document_name": "extract_exact_publication_name",
    "document_number": "extract_exact_publication_issue_number",
    "document_date": "extract_exact_publication_date_YYYY-MM-DD",
    "department_name": "extract_complete_issuing_department_ministry_name",
    "phi_themes": ["identify_and_extract_themes_topics_based_on_actual_document_content"],
    "actors_in_play": ["extract_all_named_individuals_organizations_entities"],
    "outcome_decisions": ["extract_all_explicit_decisions_rulings_determinations"],
    "outcome_reason": ["extract_all_stated_reasons_rationale_justifications"],
    "affected_parties": ["extract_all_parties_explicitly_mentioned_as_affected"],
    "acts_regs_referred": ["extract_complete_names_of_laws_regulations_acts_referenced"],
    "obligations": ["extract_all_explicitly_stated_requirements_duties_mandates"],
    "compliance_terms": ["extract_all_specific_compliance_requirements_mentioned"],
    "changes_in_acts": ["extract_explicit_amendments_modifications_to_existing_laws"],
    "key_points_of_interest": ["extract_all_significant_points_highlighted_in_document"],
    "industries_affected": ["extract_all_business_sectors_industries_specifically_mentioned"],
    "regulators_impacted": ["extract_all_regulatory_bodies_explicitly_named"],
    "fines_or_penalties": ["extract_exact_penalty_amounts_fine_structures_stated"],
    "relevant_study_type": "identify_if_clinical_trials_RWE_NIS_registries_etc",
    "status": "determine_from_list: ['Draft', 'Finalized', 'Implemented', 'New']",
    "action_required": "yes_or_no_based_on_client_relevance_and_required_actions",
    "internal_owner_stakeholder_impacted": ["extract_relevant_stakeholders_from_client_profile"],
    "government_bodies_impacted": ["extract_all_government_entities_specifically_mentioned"],
    "jurisdictions_impacted": ["extract_all_geographic_jurisdictions_explicitly_stated"],
    "regions_affected": ["extract_all_specific_regions_locations_mentioned"],
    "description": "generate_comprehensive_1000_plus_words_description_covering_all_key_aspects",
    "dates": {{
        "enforcement_date": "extract_enforcement_start_date_YYYY-MM-DD_or_None",
        "applicable_date": "extract_applicable_date_YYYY-MM-DD_or_None", 
        "comments_due_date": "extract_comments_deadline_YYYY-MM-DD_or_None",
        "guidance_issued_date": "extract_guidance_issued_date_YYYY-MM-DD_or_None",
        "expiry_date": "extract_expiry_date_YYYY-MM-DD_or_None",
        "withdrawal_date": "extract_withdrawal_date_YYYY-MM-DD_or_None",
        "extension_date": "extract_extension_date_YYYY-MM-DD_or_None",
        "publication_date": "extract_publication_date_YYYY-MM-DD_or_None",
        "effective_date": "extract_effective_date_YYYY-MM-DD_or_None",
        "exception_from_date": "extract_exception_start_date_YYYY-MM-DD_or_None",
        "exception_to_date": "extract_exception_end_date_YYYY-MM-DD_or_None",
        "due_date": "extract_general_deadline_YYYY-MM-DD_or_None",
        "compliance_due_date": "extract_compliance_deadline_YYYY-MM-DD_or_None",
        "meeting_date": "extract_meeting_hearing_date_YYYY-MM-DD_or_None",
        "hearing_date": "extract_hearing_consultation_date_YYYY-MM-DD_or_None"
    }},
    "impact_score": {{
        "outcome_decisions_impact_score": "rate_significance_of_decisions: Very_Low/Low/Moderate/High/Very_High/Critical",
        "affected_parties_impact_score": "rate_number_importance_of_affected_parties: Very_Low/Low/Moderate/High/Very_High/Critical",
        "acts_regs_referred_impact_score": "rate_importance_of_referenced_legislation: Very_Low/Low/Moderate/High/Very_High/Critical",
        "obligations_impact_score": "rate_complexity_burden_of_obligations: Very_Low/Low/Moderate/High/Very_High/Critical",
        "compliance_terms_impact_score": "rate_stringency_of_compliance_requirements: Very_Low/Low/Moderate/High/Very_High/Critical",
        "changes_in_acts_impact_score": "rate_significance_of_legislative_changes: Very_Low/Low/Moderate/High/Very_High/Critical",
        "key_points_of_interest_impact_score": "rate_importance_of_highlighted_points: Very_Low/Low/Moderate/High/Very_High/Critical",
        "industries_affected_impact_score": "rate_breadth_depth_of_industry_impact: Very_Low/Low/Moderate/High/Very_High/Critical",
        "regulators_impacted_impact_score": "rate_number_importance_of_regulators: Very_Low/Low/Moderate/High/Very_High/Critical",
        "fines_or_penalties_impact_score": "rate_severity_of_financial_penalties: Very_Low/Low/Moderate/High/Very_High/Critical",
        "government_bodies_impacted_impact_score": "rate_level_scope_of_government_involvement: Very_Low/Low/Moderate/High/Very_High/Critical",
        "jurisdictions_impacted_impact_score": "rate_geographic_scope_of_jurisdictional_impact: Very_Low/Low/Moderate/High/Very_High/Critical",
        "regions_affected_impact_score": "rate_geographic_scope_of_regional_impact: Very_Low/Low/Moderate/High/Very_High/Critical",
        "overall_impact_score": "rate_aggregate_total_impact: Very_Low/Low/Moderate/High/Very_High/Critical"
    }},
    "phi_theme_categorized": {{}},
    "blob_name": "documents/{current_date}/{pdf_filename}",
    "report": "generate_comprehensive_markdown_report_2_3_paragraphs_minimum",
    "total_token_usage": {{
        "total_tokens": 0,
        "output_tokens": 0,
        "input_tokens": 0
    }}
}}
```

## SPECIFIC EXTRACTION GUIDELINES

### Date Processing Rules
- **Enforcement Date**: Look for phrases like "comes into effect", "shall be enforced", "effective from"
- **Due Dates**: Calculate based on publication date if mentioned as "x days from publication"
- **Format Consistency**: Always convert to YYYY-MM-DD, never guess missing components
- **Missing Dates**: Use "None" if date cannot be determined with certainty

### Theme Identification
- **Natural Themes**: Identify themes based on actual document content and subject matter
- **Multiple Themes**: Include ALL relevant themes found in the document
- **Descriptive**: Use clear, descriptive theme names that reflect the document's focus
- **Comprehensive**: Don't leave theme list empty, identify at least one relevant theme

### Impact Scoring Criteria
- **Very High/Critical**: Major policy changes, significant penalties, broad impact
- **High**: Important regulatory changes, substantial requirements
- **Moderate**: Standard updates, routine compliance requirements  
- **Low**: Minor changes, limited scope
- **Very Low**: Minimal impact or procedural updates

### Action Requirements Assessment
- **Action Required "yes"**: If document requires specific actions from organizations
- **Action Required "no"**: If document is informational or not directly applicable

## QUALITY ASSURANCE CHECKLIST
Before responding, verify:
✓ All dates are in YYYY-MM-DD format or "None"
✓ All lists contain relevant items (never empty)
✓ All strings are meaningful (never empty)
✓ Document type matches one from the provided list
✓ Themes are identified from actual document content
✓ Impact scores use only the specified scale
✓ JSON structure is complete and valid

## FINAL INSTRUCTION
Extract comprehensive information from the document following the above structure exactly. Generate a complete, accurate JSON response covering all aspects of the document. Focus on accuracy and completeness while ensuring no field is left empty.

**RESPOND WITH ONLY THE VALID JSON OBJECT - NO OTHER TEXT**"""

        return prompt
    
    def _fix_output_structure(self, result: Dict[str, Any]) -> None:
        
        # Ensure required nested objects exist
        if "dates" not in result:
            result["dates"] = {}
        
        if "impact_score" not in result:
            result["impact_score"] = {}
        
        # Ensure agency field exists (fallback to department_name)
        if "agency" not in result:
            result["agency"] = result.get("department_name", "None")
        
        # Ensure _id is a string (not dict)
        if "_id" in result and isinstance(result["_id"], dict):
            if "$oid" in result["_id"]:
                notice_num = result.get("notice_number", "unknown")
                date_added = result.get("date_added", datetime.now().strftime('%Y-%m-%d'))
                result["_id"] = f"{notice_num}_{date_added}"
    
    def validate_response(self, result: Dict[str, Any]) -> bool:
        required_fields = ["_id", "unique_id", "document_type", "dates", "impact_score", "agency"]
        
        for field in required_fields:
            if field not in result:
                print(f"VALIDATION ERROR: Missing required field: {field}")
                return False
        
        print("✅ Basic validation passed")
        return True
    
    def _validate_extraction(self, extracted_text: str, initial_result: Dict[str, Any], pdf_filename: str) -> Dict[str, Any]:
        """Step 2: Senior analyst validation - identifies mistakes only using same extracted text"""
        
        validation_prompt = f"""
# SENIOR ANALYST VALIDATION CHECK

You are a **SENIOR DOCUMENT ANALYST** performing a validation check only.

## ORIGINAL DOCUMENT TEXT:
{extracted_text}

## JUNIOR ANALYST'S EXTRACTION:
{json.dumps(initial_result, indent=2)}

## YOUR TASK:
Review the junior's extraction against the original document text and identify any mistakes or missing information. 

**DO NOT** provide the corrected JSON - only identify what's wrong.

Return a JSON object with validation results in this format:

```json
{{
    "all_correct": true_or_false,
    "field_validations": {{
        "is_notice_name_correct": true_or_false,
        "is_notice_number_correct": true_or_false,
        "is_notice_date_correct": true_or_false,
        "is_agency_correct": true_or_false,
        "is_department_name_correct": true_or_false,
        "is_document_type_correct": true_or_false,
        "is_jurisdiction_correct": true_or_false,
        "is_phi_themes_correct": true_or_false,
        "is_actors_in_play_correct": true_or_false,
        "is_outcome_decisions_correct": true_or_false,
        "is_affected_parties_correct": true_or_false,
        "is_obligations_correct": true_or_false,
        "is_dates_correct": true_or_false,
        "is_description_correct": true_or_false
    }},
    "issues_found": [
        "Brief description of each issue found (if any)"
    ]
}}
```

**RESPOND WITH ONLY THE VALIDATION JSON - NO OTHER TEXT**
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior analyst performing validation checks. Identify mistakes only - do not provide corrections."
                    },
                    {
                        "role": "user", 
                        "content": validation_prompt
                    }
                ],
            )
            
            response_content = response.choices[0].message.content.strip()
            
            # Clean response
            if response_content.startswith("```json"):
                response_content = response_content[7:] 
            if response_content.startswith("```"):
                response_content = response_content[3:]   
            if response_content.endswith("```"):
                response_content = response_content[:-3]  
            
            response_content = response_content.strip()
            
            validation_result = json.loads(response_content)
            
            print(f"✅ Validation check completed - All correct: {validation_result.get('all_correct', False)}")
            return validation_result
            
        except Exception as e:
            print(f"⚠️ Validation check failed: {e}")
            return {"all_correct": True}
    
    def _correct_extraction(self, pdf_path: str, initial_result: Dict[str, Any], validation_result: Dict[str, Any], pdf_filename: str) -> Dict[str, Any]:
        """Step 3: Final correction using Gemini Flash with original PDF file"""
        
        # Check if we have valid PDF path
        if not pdf_path or not os.path.exists(pdf_path):
            print("⚠️ No valid PDF path available for Gemini correction")
            return initial_result
        
        try:
            import google.generativeai as genai
            
            # Configure Gemini (you'll need to set GOOGLE_API_KEY in your .env file)
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Upload the PDF file to Gemini
            print("Uploading PDF to Gemini Flash...")
            pdf_file = genai.upload_file(pdf_path)
            
            correction_prompt = f"""
# FINAL CORRECTION STEP

You are a **SENIOR DOCUMENT ANALYST** making final corrections using the original PDF file.

## JUNIOR ANALYST'S EXTRACTION:
{json.dumps(initial_result, indent=2)}

## VALIDATION ISSUES IDENTIFIED:
{json.dumps(validation_result, indent=2)}

## YOUR TASK:
Based on the validation issues found, correct the junior's extraction using the original PDF document. Focus only on fixing the identified problems while keeping correct information unchanged.

Return the complete corrected JSON in the same structure as the original extraction.

**RESPOND WITH ONLY THE CORRECTED JSON OBJECT - NO OTHER TEXT**
"""
            
            print("Sending correction request to Gemini Flash...")
            response = model.generate_content([pdf_file, correction_prompt])
            
            response_content = response.text.strip()
            
            # Clean response (same cleaning logic as before)
            if response_content.startswith("```json"):
                response_content = response_content[7:] 
            if response_content.startswith("```"):
                response_content = response_content[3:]   
            if response_content.endswith("```"):
                response_content = response_content[:-3]  
            
            response_content = response_content.replace(': None,', ': "None",')
            response_content = response_content.replace(': None\n', ': "None"\n')
            response_content = response_content.replace(': None}', ': "None"}')
            
            response_content = response_content.strip()
            
            corrected_result = json.loads(response_content)
            
            # Preserve token usage from initial extraction
            if "total_token_usage" in initial_result:
                corrected_result["total_token_usage"] = initial_result["total_token_usage"]
            
            self._fix_output_structure(corrected_result)
            
            print("✅ Gemini Flash correction completed")
            return corrected_result
            
        except Exception as e:
            print(f"⚠️ Gemini Flash correction failed: {e}")
            print("Returning initial result...")
            return initial_result
    
