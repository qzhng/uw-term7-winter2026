c = get_config()
c.LatexExporter.preprocessors = [
    'nbconvert.preprocessors.TagRemovePreprocessor'
]
# This removes the standard title/date block
c.LatexExporter.template_name = 'lab'