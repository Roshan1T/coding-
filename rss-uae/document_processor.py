import os
import json
import uuid
import re
import time
import random
from datetime import datetime
from typing import Dict, Any, List
from openai import AzureOpenAI
from src.logger import logger
from src.config import Config

class DocumentProcessor:
    """
    Advanced document processing with 2-stage validation approach
    Based on the proven approach from gazzete_extractor_real
    """
    
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=Config.AZ_OPENAI_API_KEY,
            api_version=os.getenv("AZ_OPENAI_API_VERSION"),
            azure_endpoint=Config.AZ_OPENAI_ENDPOINT
        )
        self.deployment_name = "gpt-4.1-mini"
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
    
    def process_document(self, extracted_text: str, themes: list, pdf_filename: str) -> List[Dict[str, Any]]:
        """Process document with 2-step validation approach"""
        
        logger.info(f"Processing document: {pdf_filename}")
        
        # Step 1: Initial extraction by junior analyst
        logger.info("Step 1: Junior analyst - Initial document analysis...")
        initial_result = self._extract_document_data(extracted_text, themes, pdf_filename)
        
        # Step 2: Senior analyst - Validation check only (using same extracted_text)
        logger.info("Step 2: Senior analyst - Validation check...")
        validation_result = self._validate_extraction(extracted_text, initial_result, pdf_filename)
        
        # Use initial result if validation passes, otherwise try to improve
        if validation_result.get("all_correct", True):
            logger.info("✅ All fields validated correctly - using initial extraction")
            return [initial_result] if not isinstance(initial_result, list) else initial_result
        else:
            logger.info("⚠️ Issues found - Attempting to improve extraction...")
            final_result = self._improve_extraction(extracted_text, initial_result, validation_result, themes, pdf_filename)
            return [final_result] if not isinstance(final_result, list) else final_result
    
    def _extract_document_data(self, extracted_text: str, themes: list, pdf_filename: str) -> Dict[str, Any]:
        """Initial extraction by junior analyst"""
        
        try:
            prompt = self.create_extraction_prompt(extracted_text, themes, pdf_filename)
            
            logger.info("Sending request to Azure OpenAI...")
            
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
            logger.info(f"Token usage - Total: {token_usage.total_tokens}, "
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
                
                # Add required fields if missing
                if "unique_id" not in result:
                    result['unique_id'] = str(uuid.uuid4())
                if "date_added" not in result:
                    result['date_added'] = datetime.today().strftime("%Y-%m-%d")
                
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Response content: {response_content}")
                raise Exception(f"Invalid JSON response from OpenAI: {e}")
                
        except Exception as e:
            logger.error(f"OpenAI processing failed: {e}")
            raise Exception(f"Failed to process document with OpenAI: {e}")

    def create_extraction_prompt(self, extracted_text: str, themes: list, pdf_filename: str) -> str:
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Format themes for the prompt
        themes_str = ""
        if themes and len(themes) > 0:
            themes_str = f"""

## AVAILABLE THEMES
Use ONLY these themes when categorizing the document content:
{', '.join(themes)}

**Important**: Only use themes from the above list that are relevant to the document content. Do not create new themes."""
        
        prompt = f"""<document>
<filename>{pdf_filename}</filename>
<content>
{extracted_text}
</content>
</document>
{themes_str}

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
- **Use Provided Themes**: ONLY use themes from the provided list above
- **Relevance Check**: Select themes that are actually relevant to the document content
- **Multiple Themes**: Include ALL relevant themes from the provided list that apply to the document
- **No New Themes**: Do not create or invent new themes - use only from the provided list
- **At Least One**: If document content matches any provided theme, include at least one relevant theme

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
            
            logger.info(f"✅ Validation check completed - All correct: {validation_result.get('all_correct', False)}")
            return validation_result
            
        except Exception as e:
            logger.error(f"⚠️ Validation check failed: {e}")
            return {"all_correct": True}
    
    def _improve_extraction(self, extracted_text: str, initial_result: Dict[str, Any], validation_result: Dict[str, Any], themes: list, pdf_filename: str) -> Dict[str, Any]:
        """Improve extraction based on validation feedback"""
        
        try:
            # Format themes for improvement prompt
            themes_str = ""
            if themes and len(themes) > 0:
                themes_str = f"\n\n**Available Themes:** {', '.join(themes)}\n**Important:** Only use themes from this list that are relevant to the document content."
            
            improvement_prompt = f"""
# EXTRACTION IMPROVEMENT

You are a **SENIOR DOCUMENT ANALYST** improving a junior analyst's work.

## ORIGINAL DOCUMENT TEXT:
{extracted_text}{themes_str}

## JUNIOR ANALYST'S EXTRACTION:
{json.dumps(initial_result, indent=2)}

## VALIDATION ISSUES IDENTIFIED:
{json.dumps(validation_result, indent=2)}

## YOUR TASK:
Based on the validation issues found, correct and improve the junior's extraction using the original document text. Focus only on fixing the identified problems while keeping correct information unchanged.

Return the complete corrected JSON in the same structure as the original extraction.

**RESPOND WITH ONLY THE CORRECTED JSON OBJECT - NO OTHER TEXT**
"""
            
            logger.info("Sending improvement request to Azure OpenAI...")
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior analyst making final corrections. Return only the improved JSON."
                    },
                    {
                        "role": "user", 
                        "content": improvement_prompt
                    }
                ],
            )
            
            response_content = response.choices[0].message.content.strip()
            
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
            
            improved_result = json.loads(response_content)
            
            # Preserve token usage from initial extraction
            if "total_token_usage" in initial_result:
                improved_result["total_token_usage"] = initial_result["total_token_usage"]
            
            self._fix_output_structure(improved_result)
            
            # Add required fields if missing
            if "unique_id" not in improved_result:
                improved_result['unique_id'] = str(uuid.uuid4())
            if "date_added" not in improved_result:
                improved_result['date_added'] = datetime.today().strftime("%Y-%m-%d")
            
            logger.info("✅ Extraction improvement completed")
            return improved_result
            
        except Exception as e:
            logger.error(f"⚠️ Extraction improvement failed: {e}")
            logger.warning("Returning initial result...")
            return initial_result
