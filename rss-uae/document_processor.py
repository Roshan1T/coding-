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
from src.themes.group_output_themes import get_grouped_themes

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
    
    def process_document(self, extracted_text: str, themes: list, rss_link: str) -> List[Dict[str, Any]]:
        """Process document with 2-step validation approach"""
        
        logger.info(f"Processing document: {rss_link}")
        
        # Step 1: Initial extraction by junior analyst
        logger.info("Step 1: Junior analyst - Initial document analysis...")
        initial_result = self._extract_document_data(extracted_text, themes, rss_link)
        
        # Step 2: Senior analyst - Validation check only (using same extracted_text)
        logger.info("Step 2: Senior analyst - Validation check...")
        validation_result = self._validate_extraction(extracted_text, initial_result, rss_link)
        
        # Use initial result if validation passes, otherwise try to improve
        if validation_result.get("all_correct", True):
            logger.info("✅ All fields validated correctly - using initial extraction")
            final_result = initial_result
        else:
            logger.info("⚠️ Issues found - Attempting to improve extraction...")
            final_result = self._improve_extraction(extracted_text, initial_result, validation_result, themes, rss_link)
        
        # Add all programmatic fields after validation is complete
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Add system fields that should not be extracted by AI
        final_result["unique_id"] = str(uuid.uuid4())
        final_result["document_type"] = "Government Gazette"
        final_result["jurisdiction"] = "United Arab Emirates"
        final_result["iso_country_code"] = "AE"
        final_result["state"] = "Federal"
        final_result["file_path"] = rss_link
        final_result["date_added"] = current_date
        final_result["language"] = "English"
        
        # Generate _id
        notice_num = final_result.get("notice_number", "unknown")
        final_result['_id'] = f"{notice_num}_{current_date}"
        
        # Handle phi_theme_categorized exactly like gpt_extraction.py
        if "None" in final_result.get("phi_themes", []):
            final_result["phi_themes"] = []
        grouped_themes = get_grouped_themes(final_result.get("phi_themes", []))
        final_result["phi_theme_categorized"] = grouped_themes
        
        # Generate blob name
        from urllib.parse import urlparse
        import os
        parsed_url = urlparse(rss_link)
        rss_filename = os.path.basename(parsed_url.path) or "document.pdf"
        final_result['blob_name'] = f"documents/{current_date}/{rss_filename}"
        
        # Handle date logic exactly like gpt_extraction.py
        if (
            final_result.get("document_date") == "None"
            and final_result.get("notice_date") != "None"
        ):
            final_result["document_date"] = final_result["notice_date"]
            logger.info("Added notice date to document date.")
        elif (
            final_result.get("notice_date") == "None"
            and final_result.get("document_date") != "None"
        ):
            final_result["notice_date"] = final_result["document_date"]
            logger.info("Added document date to notice date.")
        
        return [final_result] if not isinstance(final_result, list) else final_result
    
    def _extract_document_data(self, extracted_text: str, themes: list, rss_link: str) -> Dict[str, Any]:
        """Initial extraction by junior analyst"""
        
        try:
            prompt = self.create_extraction_prompt(extracted_text, themes, rss_link)
            
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
                max_tokens=4000
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
                # Parse the AI response first (should be an array from new prompt)
                parsed_response = json.loads(response_content)
                
                # If it's an array, take the first item; otherwise use as-is
                if isinstance(parsed_response, list) and len(parsed_response) > 0:
                    ai_extracted_data = parsed_response[0]
                else:
                    ai_extracted_data = parsed_response
                
                # Only keep AI extracted data - no programmatic fields here
                result = ai_extracted_data
                
                self._fix_output_structure(result)
                
                # Update token usage
                result["total_token_usage"] = {
                    "total_tokens": token_usage.total_tokens,
                    "output_tokens": token_usage.completion_tokens,
                    "input_tokens": token_usage.prompt_tokens
                }
                
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Response content: {response_content}")
                raise Exception(f"Invalid JSON response from OpenAI: {e}")
                
        except Exception as e:
            logger.error(f"OpenAI processing failed: {e}")
            raise Exception(f"Failed to process document with OpenAI: {e}")

    def create_extraction_prompt(self, extracted_text: str, themes: list, rss_link: str) -> str:
        
        # Format themes for the prompt - exactly like gpt_extraction.py
        themes_str = ', '.join([f'"{theme}"' for theme in themes]) if themes else ''
        
        prompt = f"""<document>
<filename>{rss_link}</filename>
<content>
{extracted_text}
</content>
</document>

1. Slowly analyze step by step the content of this Government Gazette document.
2. Extract detailed information corresponding to each notice. The output must strictly be in English language. Even if the input is in a different language, please ensure that the output is in English.
3. Identify the types of notices. Choose the correct notice_type from this list: ["Guidance", "Regulation", "Standard", "Policy", "Ministerial Decision", "Law", "Circular", "Checklist", "Framework", "General", "Resolution", "Directive", "Notification", "Order", "Decree", "Memorandum", "Bulletin", "Instruction", "Draft Guidance", "Consultation Paper", "Act", "Amendment", "Procedure", "Manual", "Protocol", "Specification", "Form", "Template", "Report", "White Paper", "Green Paper", "Charter", "Treaty", "Council Resolution", "Declaration", "Statement"].
4. Strictly retrieve and extract the values from the document. Do not add any external information. The value extracted should strictly belong to the notice. Perform step by step extraction. For each field, extract information and ensure it is verbatim from the text and do not generate any additional information.
5. For the themes, please take your time and carefully classify if the given notice falls into any of the themes in the provided list: [{themes_str}]. Please select all the themes that are related to the notice accurately. Provide the output in a list. Make sure to include all the themes that are relevant to the notice. Try to include as many as possible but make sure they are relevant to the notice. If you find multiple themes, please include all of them in the list. Do not leave the list empty. STRICTLY FOLLOW THIS OUTPUT FORMAT: ["theme1", "theme2"]
6. For enforcement_date, extract start date in which the notice or enforcement is enforced or the date of decision or the date for regulation to enter into force. Convert the date to YYYY-MM-DD format. If there is no date of enforcement or decision then keep it as "None". Make sure it is extremely accurate and correct. There might be cases where "the regulations shall come into effect after x days from the date of publication". In such cases, calculate the date by adding x in date of publication and provide the enforcement date accordingly. Instead of date of publication, it can be issuance or something else so make sure to calculate the date accordingly. If there is no clear date then keep it as "None".
7. For comments_due_date, identify and extract the submission deadline for comments. Convert the date to YYYY-MM-DD format. If the due date is contingent upon another date (e.g., publication date), calculate the due date accordingly and provide it in the specified format. If no due date is mentioned or if it cannot be determined, indicate "None." Make sure it is extremely accurate and correct.
8. Extract detailed description of the notice from the document. The description should be long and detailed capturing all the key points, dates, impacted parties and actions mentioned in the notice. The description should be at least 1000 words long and should be comprehensive.
9. Evaluate the impact of the notice across multiple dimensions and assign an appropriate impact score for each category. Use the scale: Very_Low, Low, Moderate, High, Very_High, Critical to reflect the significance and reach of the impact.
10. Provide a JSON structure with the following format for each notice and fill each keys with values and do not leave any of them empty at all. Strictly ensure that there SHOULD NOT be any empty string like "" and NOT be any empty list like []. This is a top priority. If there is no content present in the file for a key then mention "None". But do this absolutely if and only if there is no content / answer for that. Else try to extract as many details as possible.

```json
[
    {{
        "notice_name": "[SHOULD_BE_THE_FULL_TITLE_OF_THE_NOTICE_NAME_EXTRACTED_FROM_CONTENTS_PAGE]",
        "notice_number": "[THE_SPECIFIC_NUMBER_OF_THE_NOTICE]",
        "notice_date": "[SHOULD_BE_THE_NOTICE_ISSUED_DATE_CONVERTED_TO_YYYY-MM-DD_FORMAT_IT_CAN_ALSO_BE_FOUND_AT_THE_END_OF_EACH_NOTICE]",
        "notice_type": "[CHOOSE_ONE_FROM_THE_LIST_GIVEN_ABOVE_AS_PER_THE_CITY]",
        "document_name": "[NAME_OF_THE_DOCUMENT_WHERE_THE_NOTICE_IS_PUBLISHED]",
        "document_number": "[EXTRACT_THE_SPECIFIC_EDITION_OR_ISSUE_NUMBER_OF_THE_DOCUMENT]",
        "document_date": "[SHOULD_BE_THE_DOCUMENT_PUBLISHED_DATE_CONVERTED_TO_YYYY-MM-DD_FORMAT]",
        "department_name": "[THE_NAME_OF_THE_DEPARTMENT_OR_AUTHORITY_RESPONSIBLE_FOR_THE_NOTICE]",
        "phi_themes": [EXTRACT_THE_THEMES_OF_THE_NOTICE_FROM_THE_LIST_OF_THEMES_GIVEN_ABOVE],
        "actors_in_play": [EXTRACT_LIST_OF_KEY_ACTORS_OR_ENTITIES_INVOLVED_OR_MENTIONED_IN_THAT_SPECIFIC_NOTICE_ONLY],
        "outcome_decisions": [EXTRACT_DETAILED_LIST_OF_DECISIONS_AND_OUTCOMES_RESULTING_FROM_THE_NOTICE_CAPTURING_SPECIFIC_ACTIONS_RESPONSIBILITIES_AND_ROLES_OF_ENTITIES_INVOLVED_FROM_DOCUMENT_INCLUDE_ALL_NEW_POWERS_OR_NEW_DECISIONS_GIVEN_AND_OR_TAKEN_INSIDE_THE_NOTICE],
        "outcome_reason": [EXTRACT_LIST_OF_REASONS_FOR_THE_OUTCOMES_OR_DECISIONS_OF_THAT_NOTICE_FROM_DOCUMENT],
        "affected_parties": [EXTRACT_LIST_OF_PARTIES_AFFECTED_BY_THE_NOTICE_FROM_DOCUMENT],
        "acts_regs_referred": [EXTRACT_LIST_OF_ALL_LAWS_ACTS_REGULATIONS_REFERENCED_STRICTLY_WITHIN_THE_NOTICE_WITH_THEIR_COMPLETE_NAME],
        "obligations": [EXTRACT_COMPLETE_LIST_OF_THE_OBLIGATIONS_IMPOSED_BY_THE_NOTICE_INCLUDING_ALL_THE_DETAILS_OF_OBLIGATIONS_FROM_DOCUMENT],
        "compliance_terms": [EXTRACT_LIST_OF_TERMS_AND_CONDITIONS_FOR_COMPLIANCE_WITH_THE_NOTICE_FROM_DOCUMENT_DON'T_KEEP_IT_EMPTY],
        "changes_in_acts": [EXTRACT_LIST_OF_CHANGES_TO_THE_EXISTING_ACT_OR_LAW_AS_A_RESULT_OF_THE_NOTICE_FROM_THE_DOCUMENT.MENTION_THE_NAME_OF_THE_ACT_REGULATION_OR_LAW_AS_WELL_AS_WHAT_CHANGE_IS_MADE_BUT_IF_NO_CHANGE_WAS_MADE_THEN_MENTION_NONE],
        "key_points_of_interest": [EXTRACT_COMPLETE_LIST_OF_DETAILED_DESCRIPTION_OF_THE_KEY_POINTS_OF_INTEREST_ALONG_WITH_ALL_THE_DETAILS_OF_KPI_FROM_DOCUMENT],
        "industries_affected": [EXTRACT_LIST_OF_INDUSTRIES_AFFECTED_BY_THE_NOTICE_FROM_DOCUMENT],
        "regulators_impacted": [EXTRACT_LIST_OF_REGULATORS_IMPACTED_BY_THE_NOTICE_FROM_DOCUMENT],
        "fines_or_penalties": [EXTRACT_LIST_OF_FINES_OR_PENALTIES_MENTIONED_IN_THE_NOTICE_FROM_DOCUMENT],
        "government_bodies_impacted": [EXTRACT_LIST_OF_GOVERNMENT_BODIES_IMPACTED_BY_THE_NOTICE_FROM_DOCUMENT],
        "jurisdictions_impacted": [EXTRACT_LIST_OF_JURISDICTIONS_IMPACTED_BY_THE_NOTICE_FROM_DOCUMENT],
        "regions_affected": [EXTRACT_LIST_OF_REGIONS_OR_STATES_AFFECTED_BY_THE_NOTICE_FROM_DOCUMENT],
        "description": "EXTRACT_LONG_AND_DETAILED_1000_WORDS_MAKE_IT_COMPREHENSIVE_DESCRIPTION_OF_THE_NOTICE_FROM_DOCUMENT_INCLUDING_ALL_THE_KEY_POINTS_DATES_AND_IMPACTED_PARTIES_AND_ACTIONS_MENTIONED",
        "report": "GENERATE_A_COMPREHENSIVE_MARKDOWN_REPORT_2_3_PARAGRAPHS_MINIMUM_SUMMARIZING_THE_NOTICE_INCLUDING_KEY_DETAILS_IMPACT_AND_IMPLICATIONS",
        "dates": {{
            "enforcement_date": "EXTRACT_THE_START_DATE_IN_WHICH_THE_NOTICE_OR_REGULATION_IS_ENFORCED_OR_THE_DATE_OF_DECISION_OR_THE_DATE_FOR_REGULATION_TO_ENTER_INTO_FORCE_CONVERTED_TO_YYYY-MM-DD_FORMAT",
            "applicable_date": "<EXTRACT APPLICABLE DATE IN YYYY-MM-DD FORMAT IF PRESENT>",
            "comments_due_date": "<EXTRACT COMMENTS DUE DATE IN YYYY-MM-DD FORMAT IF PRESENT>",
            "guidance_issued_date": "<EXTRACT GUIDANCE ISSUED DATE IN YYYY-MM-DD FORMAT IF PRESENT>",
            "expiry_date": "<EXTRACT EXPIRY DATE IN YYYY-MM-DD FORMAT IF PRESENT>",
            "withdrawal_date": "<EXTRACT WITHDRAWAL DATE IN YYYY-MM-DD FORMAT IF PRESENT>",
            "extension_date": "<EXTRACT EXTENSION DATE IN YYYY-MM-DD FORMAT IF PRESENT>",
            "publication_date": "<EXTRACT PUBLICATION DATE IN YYYY-MM-DD FORMAT IF PRESENT>",
            "exception_from_date": "<EXTRACT EXCEPTION FROM DATE IN YYYY-MM-DD FORMAT IF PRESENT>",
            "exception_to_date": "<EXTRACT EXCEPTION TO DATE IN YYYY-MM-DD FORMAT IF PRESENT>",
            "due_date": "<EXTRACT DUE DATE IN YYYY-MM-DD FORMAT IF PRESENT>",
            "compliance_due_date": "<EXTRACT COMPLIANCE DUE DATE IN YYYY-MM-DD FORMAT IF PRESENT>",
            "meeting_date": "<EXTRACT MEETING DATE IN YYYY-MM-DD FORMAT IF PRESENT>",
            "hearing_date": "<EXTRACT HEARING DATE IN YYYY-MM-DD FORMAT IF PRESENT>",
            "effective_date": "<EXTRACT EFFECTIVE DATE IN YYYY-MM-DD FORMAT IF PRESENT>"
        }},
        "impact_score": {{
            "outcome_decisions_impact_score": "Give a impact score based on: How significant are the decisions made? Do they set precedents or have wide-reaching consequences?",
            "affected_parties_impact_score": "Give a impact score based on: How many individuals, organizations, or sectors are impacted?",
            "acts_regs_referred_impact_score": "Give a impact score based on: Do the referenced laws introduce major changes or establish new legal precedents?",
            "obligations_impact_score": "Give a impact score based on: Are there significant legal, financial, or operational burdens introduced?",
            "compliance_terms_impact_score": "Give a impact score based on: How strict and immediate are the compliance requirements?",
            "changes_in_acts_impact_score": "Give a impact score based on: Are existing laws being amended in a way that alters regulations or enforcement?",
            "key_points_of_interest_impact_score": "Give a impact score based on: Do the highlighted issues indicate major shifts or critical concerns?",
            "industries_affected_impact_score": "Give a impact score based on: How many sectors or industries are impacted?",
            "regulators_impacted_impact_score": "Give a impact score based on: Do multiple agencies or governing bodies need to respond or adapt?",
            "fines_or_penalties_impact_score": "Give a impact score based on: Are there significant financial penalties or legal consequences?",
            "government_bodies_impacted_impact_score": "Give a impact score based on: Are key government agencies or departments affected?",
            "regions_impacted_impact_score": "Give a impact score based on: Is the impact localized or does it span multiple regions or countries?",
            "overall_impact_score": "Give a impact score : Very Low, Low, Moderate, High, Very High, Critical  based on the overall impact and significance of the notice"
        }}
    }}
]
```

Guidelines:
- Strictly maintain the above format.
- Keep accuracy and completeness as first priority. Even if there is extensive length and detailed nature required, extract all the details.
- Ensure that no value or list is left empty. All the keys must have non empty strings or non-empty lists.
- Each notice should have its own dictionary.
- The data should be extracted strictly from that particular notice only. Never extract data from other notices.
- Properly identify the notice name and extract it exactly as present in the document.
- Ensure that acts, laws, regulations are placed correctly to the notice that they are mentioned in.
- Ensure as much information as possible is extracted from the document. Extract the complete list of laws, acts, regulations that is being referred to.
- Extract content strictly from the provided document without any modifications or external additions.
- For key points of interest, include a description and a complete list of all the given items from the documents.
- If the value contains reference to additional information, make sure they are extracted from the document too.
- Strictly follow the given output format. Don't add any comments.
- The response should be in English language only.
- In the acts_regs_referred value, include all the laws, acts, regulations mentioned in the notice along with their full name not just their numbers.

**RESPOND WITH ONLY THE VALID JSON ARRAY - NO OTHER TEXT**"""

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

    def _validate_extraction(self, extracted_text: str, initial_result: Dict[str, Any], rss_link: str) -> Dict[str, Any]:
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
        "is_department_name_correct": true_or_false,
        "is_notice_type_correct": true_or_false,
        "is_document_name_correct": true_or_false,
        "is_document_date_correct": true_or_false,
        "is_phi_themes_correct": true_or_false,
        "is_actors_in_play_correct": true_or_false,
        "is_outcome_decisions_correct": true_or_false,
        "is_affected_parties_correct": true_or_false,
        "is_obligations_correct": true_or_false,
        "is_dates_correct": true_or_false,
        "is_description_correct": true_or_false,
        "is_report_correct": true_or_false
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
                max_tokens=2000
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
    
    def _improve_extraction(self, extracted_text: str, initial_result: Dict[str, Any], validation_result: Dict[str, Any], themes: list, rss_link: str) -> Dict[str, Any]:
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
                max_tokens=4000
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
