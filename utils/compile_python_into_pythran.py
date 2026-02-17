# build.py
# =========================================================
# Usage:
#   python build.py -s training_loop.py
#
# It concatenates: [required_files from config_for_compilation(), then the main file]
# You can still add #include "file.py" lines inside either file; they'll be expanded.

import re
import argparse
import time
import os
import shlex
import sysconfig
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
from types import ModuleType

include_pattern = re.compile(r'^\s*#include\s+"([^"]+)"\s*$')

def expand_file(path: Path) -> str:
    out_lines = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            m = include_pattern.match(line)
            if m:
                inc_path = Path(m.group(1))
                if not inc_path.exists():
                    raise FileNotFoundError(f'Included file not found: {inc_path}')
                out_lines.append(expand_file(inc_path))
                out_lines.append('\n')
            else:
                out_lines.append(line)
    return ''.join(out_lines)


def load_function_from_path(file_path, name_of_function="config_for_compilation"):
    p = Path(file_path)

    if not p.exists():
        raise ValueError(f"File does not exist: {p}")
    if p.suffix != ".py":
        raise ValueError(f"Expected a .py file, got: {p.name}")

    module_name = f"_dyn_{p.stem}_{abs(hash(p.resolve()))}"

    spec = spec_from_file_location(module_name, p)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not create a module spec for {p}")

    mod: ModuleType = module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # executes the module code
    except Exception as e:
        raise ValueError(f"Error executing module {p}: {e}") from e

    try:
        func = getattr(mod, name_of_function)
    except AttributeError as e:
        raise ValueError(
            f"{name_of_function} not found in {p}. Define a top-level function named {name_of_function}."
        ) from e

    if not callable(func):
        raise ValueError(f"{name_of_function} in {p} is not callable.")

    return func

def create_combined_file(main_path: Path, combined_file_path: Path):
    # ensure parent dir exists
    combined_file_path.parent.mkdir(parents=True, exist_ok=True)

    root_path = main_path.parent
    config_func = load_function_from_path(main_path)
    config = config_func()
    list_of_required_files = [root_path / file for file in config.get('required_files', [])]

    combined = []
    combined.append("# ---- BEGIN Required Files ----\n")
    for required_file in list_of_required_files:
        combined.append(expand_file(Path(required_file)))
    combined.append("\n# ---- END Required Files ----\n\n")
    combined.append("# ---- BEGIN MAIN / TRAINING LOOP ----\n")
    combined.append(expand_file(main_path))
    combined.append("\n# ---- END MAIN / TRAINING LOOP ----\n")

    combined_file_path.write_text(''.join(combined), encoding='utf-8')
    print(f"Successfully created {combined_file_path}")


def compile_into_pythran(source_file: str):
    main_path = Path(source_file).resolve()
    out_dir = main_path.parent / "_compiled_files"
    combined_file_path = out_dir / f"{main_path.stem}___ready_to_compile.py"

    # combine
    create_combined_file(main_path, combined_file_path)

    # compile
    ext = sysconfig.get_config_var('EXT_SUFFIX') or '.so'
    compiled_path = (out_dir / main_path.stem).with_suffix(ext)
    os.system(f"rm -f {out_dir / main_path.stem}*.so") # remove the old so file

    #cmd = f"pythran -O3 -march=native -o {shlex.quote(str(compiled_path))} {shlex.quote(str(combined_file_path))}"
    cmd = f"pythran -O2 -o {shlex.quote(str(compiled_path))} {shlex.quote(str(combined_file_path))}"
    print(f"Running: {cmd}")
    compilation_start_time = time.time()
    ret = os.system(cmd)
    print(f'Compilation took {int(time.time()-compilation_start_time)} sec')
    if ret != 0:
        raise SystemExit(f"Pythran compilation failed with exit code {ret}")
    print(f"Built {compiled_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--source_file', required=True, help='Main/training loop file (e.g., training_loop.py)')
    args = ap.parse_args()

    compile_into_pythran(args.source_file)
