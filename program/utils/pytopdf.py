import os
import glob
from fpdf import FPDF

def crear_pdf_desde_py():
    # Buscar todos los archivos .py en el directorio actual y subdirectorios
    archivos_py = glob.glob('**/*.py', recursive=True)
    
    # Crear un nuevo PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Iterar sobre cada archivo .py encontrado
    for archivo in archivos_py:
        # Añadir el nombre del archivo como título
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt=archivo, ln=1, align='L')
        pdf.set_font("Arial", size=12)
        
        # Leer el contenido del archivo
        with open(archivo, 'r', encoding='utf-8') as f:
            contenido = f.read()
        
        # Añadir el contenido al PDF
        pdf.multi_cell(0, 10, txt=contenido)
        
        # Añadir un salto de página después de cada archivo
        pdf.add_page()
    
    # Guardar el PDF
    pdf.output("codigo_python_combinado.pdf")

if __name__ == "__main__":
    crear_pdf_desde_py()
