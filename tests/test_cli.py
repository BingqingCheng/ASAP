"""
Testing the CLI
"""
import os

from click.testing import CliRunner

from asaplib.cli.cmd_asap import asap


def test_cmd_gen_soap():
    """Test the command for generating soap descriptors"""
    test_folder = os.path.split(__file__)[0]
    xyzpath = os.path.abspath(os.path.join(test_folder, 'small_molecules-1000.xyz'))
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(asap, ['gen_desc', '--fxyz', xyzpath, 'soap'])
    assert result.exit_code == 0
    assert 'Using SOAP' in result.stdout
