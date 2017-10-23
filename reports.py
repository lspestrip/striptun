# -*- encoding: utf-8 -*-

from datetime import datetime
import logging as log
import os
from shutil import copyfile
import subprocess
from typing import Any, Dict

from mako.template import Template
from markdown import markdown

from version import VERSION


def get_latest_git_commit(error_value='<unknown>') -> str:
    '''Return a string containing the hash of the latest commit.

    This procedure assumes that Git is available. If any error happens, the string "error_value" will be returned.
    '''
    try:
        result = subprocess.run([
            'git',
            '-C', os.path.dirname(os.path.abspath(__file__)),
            'log',
            '-n', '1',
            '--pretty=format:"%H"'
        ], shell=True, stdout=subprocess.PIPE)

        # The [1:-1] are needed to strip the quotation marks
        return result.stdout.decode('utf-8').strip()[1:-1]
    except subprocess.CalledProcessError:
        return error_value


def create_report(params: Dict[str, Any],
                  md_template_file: str,
                  md_report_file: str,
                  html_report_file: str,
                  output_path: str):
    '''Saves a report of the tuning in the output path.

    This function assumes that ``output_path`` points to a directory that already exists.
    '''

    template_path = os.path.join(os.path.dirname(__file__), 'template')

    # Copy all the static files into the destination directory
    for static_file_name in ['report_style.css']:
        copyfile(os.path.join(template_path, static_file_name),
                 os.path.join(output_path, static_file_name))

    # Load the file containing the Markdown template in a string
    template_file_name = os.path.join(template_path, md_template_file)
    log.info('Reading report template from "%s"', template_file_name)
    report_template = Template(filename=template_file_name)

    # Fill the template and save the report in Markdown format
    extended_params = dict(params, **{
        'striptun_version': VERSION,
        'date': datetime.now().strftime('%d %b %Y, %H:%M:%S'),
        'latest_git_commit': get_latest_git_commit(),
    })
    md_report = report_template.render_unicode(**extended_params)
    md_report_path = os.path.join(output_path, md_report_file)
    with open(md_report_path, 'wt', encoding='utf-8') as md_file:
        md_file.write(md_report)
    log.info('Markdown report saved to "%s"', md_report_path)

    # Convert the report to HTML and save it too
    html_report = '''<!DOCTYPE html>
<html>
    <head>
        <title>{title}</title>
        <meta charset="UTF-8">
        <link rel="stylesheet" href="report_style.css" type="text/css" />
    </head>
    <body>
        <div id="main">
{contents}
        </div>
    </body>
</html>
'''.format(title=params['title'],
           contents=markdown(md_report, extensions=[
               'markdown.extensions.attr_list',
               'markdown.extensions.tables',
               'markdown.extensions.toc']
    ))

    html_report_path = os.path.join(output_path, html_report_file)
    with open(html_report_path, 'wt', encoding='utf-8') as html_file:
        html_file.write(html_report)
    log.info('HTML report saved to "%s"', html_report_path)
