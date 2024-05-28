set CONAN_EXE=conan.exe

set BUILD_TYPE=Release

%CONAN_EXE% profile detect --force 

%CONAN_EXE% install ./win32_conanfile.txt --build=missing  --settings:all=compiler.cppstd=17 --settings:all=build_type=%BUILD_TYPE%  --options=onnx/*:disable_static_registration=True
cmake --preset conan-default
cd build/
cmake --build . --config %BUILD_TYPE%
@REM cmake --install .
cd ..