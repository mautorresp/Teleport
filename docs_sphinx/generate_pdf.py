#!/usr/bin/env python3
"""
Generate comprehensive PDF documentation for Teleport CLF Calculator.
Combines all HTML pages into a single, complete PDF document.
"""

import weasyprint
from pathlib import Path
import os

def generate_comprehensive_pdf():
    """Generate comprehensive PDF from all documentation pages."""
    
    # Base paths
    docs_dir = Path('/Users/Admin/Teleport/docs_sphinx')
    html_dir = docs_dir / '_build/html'
    
    # All documentation pages in logical order
    pages = [
        'index.html',
        'quickstart.html', 
        'mathematical_foundation.html',
        'api_reference.html',
        'examples.html',
        'testing.html',
        'clf_calculator.html',
        'clf_maximal_validator.html',
        'docstring_guide.html',
        'changelog.html',
        'contributing.html'
    ]
    
    print("Generating comprehensive Teleport CLF Calculator PDF documentation...")
    
    # Create HTML content that combines all pages
    combined_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Teleport CLF Calculator - Complete Documentation</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }
            .math { font-family: 'Times New Roman', serif; font-style: italic; }
            code { background: #f8f9fa; padding: 2px 4px; border-radius: 3px; font-family: 'Courier New', monospace; }
            pre { background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }
            .page-break { page-break-before: always; }
            .note { background: #e8f4fd; border-left: 4px solid #3498db; padding: 10px; margin: 10px 0; }
        </style>
    </head>
    <body>
    """
    
    # Add title page
    combined_html += """
    <div style="text-align: center; margin-top: 100px;">
        <h1 style="font-size: 36px; color: #2c3e50;">Teleport CLF Calculator</h1>
        <h2 style="font-size: 24px; color: #7f8c8d;">Complete Documentation</h2>
        <div style="margin-top: 50px;">
            <p><strong>Mathematical Contract:</strong> C<sub>min</sub><sup>(1)</sup>(L) = 88 + 8×leb(L)</p>
            <p><strong>Decision Rule:</strong> EMIT ⟺ C<sub>min</sub><sup>(1)</sup>(L) < 10×L (strict)</p>
            <p><strong>Constants:</strong> H=56, CAUS=27, END=5 (locked)</p>
        </div>
        <div style="margin-top: 100px;">
            <p>Generated: September 23, 2025</p>
            <p>Version: 1.0.0 Professional Release</p>
        </div>
    </div>
    <div class="page-break"></div>
    """
    
    # Process each page
    for i, page in enumerate(pages):
        html_file = html_dir / page
        
        if not html_file.exists():
            print(f"Warning: {page} not found, skipping...")
            continue
            
        print(f"Processing {page}...")
        
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract main content (remove navigation, etc.)
            # Find content between <div class="document"> and </div>
            start = content.find('<div class="document">')
            end = content.find('</div>', start)
            
            if start != -1 and end != -1:
                main_content = content[start:end + 6]
                # Clean up navigation elements
                main_content = main_content.replace('<div class="sphinxsidebar"', '<div style="display:none" class="sphinxsidebar"')
                main_content = main_content.replace('href="#', 'href="javascript:void(0)#')
                
                # Add page break except for first page
                if i > 0:
                    combined_html += '<div class="page-break"></div>\n'
                
                combined_html += main_content + '\n\n'
            
        except Exception as e:
            print(f"Error processing {page}: {e}")
            # Add simple fallback content
            combined_html += f'<div class="page-break"></div>\n<h1>Error loading {page}</h1>\n<p>{e}</p>\n\n'
    
    # Close HTML
    combined_html += """
    </body>
    </html>
    """
    
    # Write combined HTML temporarily
    temp_html = docs_dir / 'temp_combined.html'
    with open(temp_html, 'w', encoding='utf-8') as f:
        f.write(combined_html)
    
    # Generate PDF
    output_pdf = docs_dir / 'Teleport_CLF_Calculator_Complete_Documentation.pdf'
    
    try:
        print("Converting to PDF...")
        weasyprint.HTML(filename=str(temp_html)).write_pdf(
            str(output_pdf),
            stylesheets=[],
        )
        print(f"SUCCESS: Complete PDF generated: {output_pdf}")
        
        # Clean up temp file
        temp_html.unlink()
        
        # Show file info
        file_size = output_pdf.stat().st_size
        print(f"File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        
    except Exception as e:
        print(f"ERROR: Error generating PDF: {e}")
        if temp_html.exists():
            temp_html.unlink()

if __name__ == "__main__":
    generate_comprehensive_pdf()