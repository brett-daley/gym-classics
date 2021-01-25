from importlib import import_module
from inspect import cleandoc
import os
import re
import sys

import gym

# Make sure we import the source code and not the installed version
sys.path.insert(0, '.')
import gym_classics


def main():
    env_specs = tuple(s for s in gym.envs.registry.all() if 'gym_classics' in s.entry_point)
    assert env_specs

    # Pull the docstring from each environment class to build the glossary    
    glossary = ''
    for s in env_specs:\
        # Get the environment class
        path, name = s.entry_point.split(':')
        module = import_module(path)
        env_cls = getattr(module, name)

        # Convert the docstring to a markdown table entry
        description = cleandoc(env_cls.__doc__)
        description = description.rstrip().replace('\n', '<br>')
        glossary += f"| `{s.id}` | {description} |\n"

    # Load the template
    template_path = os.path.join('readme', 'TEMPLATE.md')
    with open(template_path, 'r') as f:
        template = f.read()

    # Fill in the template
    template = template.replace('GLOSSARY', glossary)

    # Auto-cite
    pattern = re.compile('cite\{([0-9]+)\}')
    while True:
        match = pattern.search(template)
        if not match:
            break
        n = match.groups()[0]
        template = pattern.sub(f"[[{n}]](#references)", template)

    # Write to the readme
    with open('README.md', 'w') as f:        
        f.write(template)


if __name__ == '__main__':
    main()
