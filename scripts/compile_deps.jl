using PackageCompiler


create_sysimage(:MOT, sysimage_path="/project/deps.so",
                precompile_execution_file="/project/test/runtests.jl");
