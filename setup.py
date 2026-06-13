import os
import sys
import platform
import subprocess
import shutil
from setuptools import setup
from setuptools.command.build_py import build_py

class CustomBuildPy(build_py):
    def run(self):
        super().run()

        src_ext_dir   = os.path.abspath(os.path.join(os.path.dirname(__file__), "src", "basicdt", "_ext"))
        build_ext_dir = os.path.abspath(os.path.join(self.build_lib, "basicdt", "_ext"))

        system = platform.system()
        if system == "Darwin":
            lib_name = "libbasicdt.dylib"
            # NOTE: do NOT use -ffast-math. It implies -ffinite-math-only, which makes
            # std::isnan() fold to constant false and silently breaks all missing-value
            # handling. -fno-math-errno keeps the SIMD speedup without touching NaN/Inf.
            cmd = ["clang++", "-O3", "-march=native", "-fno-math-errno", "-funroll-loops", "-shared", "-fPIC", "-std=c++17",
                   "-framework", "Accelerate"]
            try:
                omp = subprocess.check_output(["brew", "--prefix", "libomp"], stderr=subprocess.DEVNULL).decode().strip()
                cmd += ["-Xpreprocessor", "-fopenmp", f"-I{omp}/include", f"-L{omp}/lib", "-lomp"]
            except Exception:
                pass
        elif system == "Windows":
            lib_name = "basicdt.dll"
            # /fp:fast can break NaN semantics; /fp:precise keeps std::isnan correct.
            cmd = ["cl", "/O2", "/fp:precise", "/LD", "/EHsc", "/openmp"]
        else:
            lib_name = "libbasicdt.so"
            # See Darwin note: -ffast-math breaks std::isnan. -fno-math-errno is safe.
            cmd = ["g++", "-O3", "-march=native", "-fno-math-errno", "-funroll-loops", "-shared", "-fPIC", "-std=c++17", "-fopenmp"]

        src_path = os.path.join(src_ext_dir, "basicdt.cpp")
        src_lib  = os.path.join(src_ext_dir, lib_name)
        build_lib_path = os.path.join(build_ext_dir, lib_name)

        if os.path.exists(src_path):
            compile_cmd = cmd + [src_path] + ["-o", src_lib]
            print(f"Compiling C++ extension: {' '.join(compile_cmd)}")
            subprocess.run(compile_cmd, check=True)

            os.makedirs(build_ext_dir, exist_ok=True)
            print(f"Copying compiled library to build folder: {build_lib_path}")
            shutil.copy2(src_lib, build_lib_path)

            for src_name in ["basicdt.cpp", "basicdt_types.h", "basicdt_core.h"]:
                src_file_in_build = os.path.join(build_ext_dir, src_name)
                if os.path.exists(src_file_in_build):
                    print(f"Removing source file {src_file_in_build} from build folder")
                    os.remove(src_file_in_build)
        else:
            print("C++ source files not found in source directory. Skipping compilation.")

setup(
    cmdclass={"build_py": CustomBuildPy},
)
