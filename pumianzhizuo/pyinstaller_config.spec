# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(['项目结构示例.py'],
             pathex=[],
             binaries=[],
             datas=[],
             hiddenimports=['sklearn.neighbors._typedefs',
                          'sklearn.utils._cython_blas',
                          'sklearn.neighbors._quad_tree',
                          'sklearn.tree._utils',
                          'sklearn.utils._typedefs',
                          'librosa',
                          'soundfile',
                          'numpy'],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

# 添加资源文件
a.datas += Tree('./resources', prefix='resources')

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='音乐游戏谱面生成器',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None,
          icon='resources/icon.ico')

# 创建Windows安装程序
if platform.system() == 'Windows':
    coll = COLLECT(exe,
                  a.binaries,
                  a.zipfiles,
                  a.datas,
                  strip=False,
                  upx=True,
                  upx_exclude=[],
                  name='音乐游戏谱面生成器') 