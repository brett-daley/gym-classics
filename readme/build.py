from importlib import import_module
from inspect import cleandoc
import itertools
import os
import re
import sys

import gym

# Make sure we import the source code and not the installed version
sys.path.insert(0, '.')
import gym_classics
gym_classics.register('gym')


def main():
    env_specs = []
    for env_id, spec in gym.envs.registry.items():
        if 'gym_classics' in spec.entry_point:
            env_specs.append(spec)
    assert env_specs

    # Pull the docstring from each environment class to build the glossary    
    glossary = ''
    for i, spec in enumerate(env_specs):
        # Get the environment class
        path, name = spec.entry_point.split(':')
        module = import_module(path)
        env_cls = getattr(module, name)

        # Convert the docstring to a markdown table entry
        description = cleandoc(env_cls.__doc__)
        description = description.rstrip()
        description = description.replace('\n\n', '<br><br>')
        description = description.replace('\n', ' ')
        glossary += f"| {i+1} | `{spec.id}` | {description} |\n"

    # Load the template
    template_path = os.path.join('readme', 'TEMPLATE.md')
    with open(template_path, 'r') as f:
        template = f.read()

    # Fill in the template
    template = template.replace('GLOSSARY', glossary)

    # Auto-cite
    for i in itertools.count(start=1):
        pattern = re.compile(r'cite\{' + str(i) + r'\}')
        match = pattern.search(template)
        if not match:
            break
        template = pattern.sub(f"[[{i}]](#references)", template)

    # Import python code
    pattern = re.compile(r'PYTHON{(.*?)}')
    for match in re.finditer(pattern, template):
        import_statement = match.group(0)
        code_file = match.group(1)
        with open(code_file, 'r') as f:
            code = f.read()
        template = template.replace(import_statement, '```python\n' + code + '```')

    # Write to the readme
    with open('README.md', 'w') as f:        
        f.write(template)


if __name__ == '__main__':
    main()
