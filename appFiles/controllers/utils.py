

def load_base_prompts(db_manager):
        system_prompt = db_manager.get_latest_base_prompts()
        if system_prompt:
            return system_prompt
        return {}