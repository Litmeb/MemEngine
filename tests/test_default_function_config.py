import unittest

from default_config.DefaultFunctionConfig import DEFAULT_IMPORTANCE_JUDGE


class DefaultImportancePromptTemplateTests(unittest.TestCase):
    def test_default_importance_prompt_escapes_literal_json_braces(self):
        template = DEFAULT_IMPORTANCE_JUDGE['prompt']['template']

        rendered = template.format(message='Remember to buy milk.')

        self.assertIn('{"score": 7}', rendered)
        self.assertNotIn('{{"score": 7}}', rendered)


if __name__ == '__main__':
    unittest.main()
