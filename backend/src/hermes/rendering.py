

def render_main_log_html(emergency_columns, transcript_columns, 
        emergency_rows, transcript_rows):
    # Gera HTML para emergencies
    emergency_html = "<h2>Emergencies</h2><table border='1'><tr>"
    for col in emergency_columns:
        emergency_html += f"<th>{col}</th>"
    emergency_html += "</tr>"
    for row in emergency_rows:
        emergency_html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    emergency_html += "</table>"

    # Gera HTML para emergency_transcripts
    transcript_html = "<h2>Emergency Transcripts</h2><table border='1'><tr>"
    for col in transcript_columns:
        transcript_html += f"<th>{col}</th>"
    transcript_html += "</tr>"
    for row in transcript_rows:
        transcript_html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    transcript_html += "</table>"

    log_html = emergency_html + "<br>" + transcript_html
    return log_html

# ...existing code...
def render_main_log_html2(emergency_columns, transcript_columns, emergency_rows, transcript_rows, audio_columns, audio_rows):
    # CSS simples para visual mais agradável
    style = """
    <style>
        table { border-collapse: collapse; width: 90%; margin-bottom: 30px; }
        th, td { border: 1px solid #aaa; padding: 6px 12px; text-align: left; }
        th { background: #f0f0f0; }
        h2 { color: #2c3e50; margin-top: 30px; }
        tr:nth-child(even) { background: #f9f9f9; }
        body { font-family: Arial, sans-serif; background: #fafbfc; }
        .container { max-width: 1200px; margin: 0 auto; }
    </style>
    <div class="container">
    """

    # Emergencies
    emergency_html = "<h2>Emergencies</h2><table><tr>"
    for col in emergency_columns:
        emergency_html += f"<th>{col}</th>"
    emergency_html += "</tr>"
    for row in emergency_rows:
        emergency_html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    emergency_html += "</table>"

    # Emergency Transcripts
    transcript_html = "<h2>Emergency Transcripts</h2>"
    for i, row in enumerate(transcript_rows):
        transcript_html += f"<div style='border: 1px solid #ccc; margin: 10px 0; padding: 15px; background: #f9f9f9;'>"
        
        # Metadata row
        transcript_html += "<div style='display: flex; gap: 20px; margin-bottom: 10px; font-size: 12px; color: #666;'>"
        for j, cell in enumerate(row):
            if transcript_columns[j] != 'part':
                transcript_html += f"<span><strong>{transcript_columns[j]}:</strong> {cell}</span>"
        transcript_html += "</div>"
        
        # Transcription text
        part_index = transcript_columns.index('part') if 'part' in transcript_columns else -1
        if part_index >= 0 and part_index < len(row):
            transcript_html += f"<div style='background: white; padding: 10px; border-radius: 4px; white-space: pre-wrap;'>{row[part_index]}</div>"
        
        transcript_html += "</div>"

    # Emergency Audios
    audio_html = "<h2>Emergency Audios</h2><table><tr>"
    for col in audio_columns:
        audio_html += f"<th>{col}</th>"
    audio_html += "</tr>"
    for row in audio_rows:
        audio_html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    audio_html += "</table>"

    # Junta tudo e fecha o container
    log_html = style + emergency_html + transcript_html + audio_html + "</div>"
    return log_html

def render_main_log_html3(emergency_columns, transcript_columns, emergency_rows, 
                          transcript_rows, audio_columns, audio_rows, 
                          inference_columns, inference_rows):
    # CSS simples para visual mais agradável
    style = """
    <style>
        table { border-collapse: collapse; width: 90%; margin-bottom: 30px; }
        th, td { border: 1px solid #aaa; padding: 6px 12px; text-align: left; }
        th { background: #f0f0f0; }
        h2 { color: #2c3e50; margin-top: 30px; }
        tr:nth-child(even) { background: #f9f9f9; }
        body { font-family: Arial, sans-serif; background: #fafbfc; }
        .container { max-width: 1200px; margin: 0 auto; }
    </style>
    <div class="container">
    """

    # Emergencies
    emergency_html = "<h2>Emergencies</h2><table><tr>"
    for col in emergency_columns:
        emergency_html += f"<th>{col}</th>"
    emergency_html += "</tr>"
    for row in emergency_rows:
        emergency_html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    emergency_html += "</table>"

    # Emergency Transcripts
    transcript_html = "<h2>Emergency Transcripts</h2>"
    for i, row in enumerate(transcript_rows):
        transcript_html += f"<div style='border: 1px solid #ccc; margin: 10px 0; padding: 15px; background: #f9f9f9;'>"
        
        # Metadata row
        transcript_html += "<div style='display: flex; gap: 20px; margin-bottom: 10px; font-size: 12px; color: #666;'>"
        for j, cell in enumerate(row):
            if transcript_columns[j] != 'part':
                transcript_html += f"<span><strong>{transcript_columns[j]}:</strong> {cell}</span>"
        transcript_html += "</div>"
        
        # Transcription text
        part_index = transcript_columns.index('part') if 'part' in transcript_columns else -1
        if part_index >= 0 and part_index < len(row):
            transcript_html += f"<div style='background: white; padding: 10px; border-radius: 4px; white-space: pre-wrap;'>{row[part_index]}</div>"
        
        transcript_html += "</div>"

    # Emergency Audios
    audio_html = "<h2>Emergency Audios</h2><table><tr>"
    for col in audio_columns:
        audio_html += f"<th>{col}</th>"
    audio_html += "</tr>"
    for row in audio_rows:
        audio_html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    audio_html += "</table>"

    # Results of Inference
    inference_html = "<h2>Results of Inference</h2>"
    for i, row in enumerate(inference_rows):
        inference_html += f"<div style='border: 1px solid #ccc; margin: 10px 0; padding: 15px; background: #f9f9f9;'>"
        
        # Metadata row
        inference_html += "<div style='display: flex; gap: 20px; margin-bottom: 10px; font-size: 12px; color: #666;'>"
        for j, cell in enumerate(row):
            if inference_columns[j] != 'resultado':
                inference_html += f"<span><strong>{inference_columns[j]}:</strong> {cell}</span>"
        inference_html += "</div>"
        
        # Result text
        result_index = inference_columns.index('resultado') if 'resultado' in inference_columns else -1
        if result_index >= 0 and result_index < len(row):
            inference_html += f"<div style='background: white; padding: 10px; border-radius: 4px; white-space: pre-wrap;'>{row[result_index]}</div>"
        
        inference_html += "</div>"

    # Junta tudo e fecha o container
    log_html = style + emergency_html + transcript_html + audio_html + inference_html + "</div>"
    return log_html