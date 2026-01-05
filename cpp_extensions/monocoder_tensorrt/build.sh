#!/bin/bash
echo "=================================="
echo "🔨 Compilando MonoCoder TRT C++"
echo "=================================="

# Activar entorno virtual
source ../../ompar_env/bin/activate

# Limpiar build anterior
rm -rf build
mkdir -p build
cd build

echo "📁 Configurando CMake..."
cmake ..

echo "🔧 Compilando..."
make -j$(nproc)

echo "✅ Instalando..."
make install

echo "=================================="
if [ -f "../../monocoder_trt_cpp.so" ]; then
    echo "🎉 Build exitoso! Módulo creado en ../../monocoder_trt_cpp.so"
else
    echo "❌ Build fallido"
    exit 1
fi
