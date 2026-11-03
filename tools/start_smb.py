from impacket import smbserver
import sys

def main():
    server = smbserver.SimpleSMBServer('0.0.0.0', 8445)
    server.setSMB2Support(True)
    server.addShare('models', '/home/rover/models', 'Models Share', readOnly='yes')
    print("SMB_SERVER_READY", flush=True)
    server.start()

if __name__ == '__main__':
    main()
