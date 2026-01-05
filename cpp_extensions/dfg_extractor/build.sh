#!/bin/bash
# Script de compilación para DFG Extractor C++
# Paso 1.4: Compilación

set -e  # Exit on error

echo "=================================="
echo "🔨 Compilando DFG Extractor C++"
echo "=================================="

# Directorio actual
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Crear directorio de build
echo "📁 Creando directorio de build..."
mkdir -p build
cd build

# Configurar con CMake
echo "⚙️  Configurando con CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -fopenmp"

# Compilar
echo "🔧 Compilando..."
make -j$(nproc)

# Instalar
echo "📦 Instalando módulo..."
make install

echo ""
echo "=================================="
echo "✅ Compilación exitosa!"
echo "=================================="
echo ""
echo "El módulo 'dfg_extractor_cpp.so' está disponible en:"
echo "  $(realpath ../../dfg_extractor_cpp.so)"
echo ""
echo "Para probar:"
echo "  cd ../.."
echo "  python3 -c 'import dfg_extractor_cpp; print(dfg_extractor_cpp.__doc__)'"
echo ""
