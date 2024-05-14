AddOption('--dbg', action='append_const', dest='cflags', const='-g')
AddOption('--opt', action='append_const', dest='cflags', const='-Os')

import os
common_env=Environment(ENV=os.environ)

acenet = ARGUMENTS.get('acenet', 0)
ccanada = ARGUMENTS.get('ccanada', 0)
think = ARGUMENTS.get('think', 0)

common_env.Replace(CXX='mpic++')

if int(acenet):
    common_env.Replace(CXX='/usr/local/gcc-4.8.3/bin/g++')
    common_env.Replace(CC='/usr/local/gcc-4.8.3/bin/gcc')
    common_env.Append(CPPDEFINES=['NOSDL'])

if int(ccanada):
    common_env.Append(CPPDEFINES=['CCANADA'])
else:
    common_env.Append(CCFLAGS = ['-std=c++20', '-Wno-deprecated', '-Wall', '-Werror', '-Wextra', '-Wno-unused-parameter'])
# '-DARMA_USE_BLAS', '-DARMA_USE_LAPACK', '-DARMA_USE_HDF5' '-DARMA_DONT_USE_WRAPPER

common_env.MergeFlags(GetOption('cflags'))
common_env.Append(CPPDEFINES={'VERSION': 1})

# Our release build is derived from the common build environment...
release_env = common_env.Clone()
# ... and adds a RELEASE preprocessor symbol ...
release_env.Append(CPPDEFINES=['RELEASE'])
# ... and release builds end up in the "build/release" dir
release_env.VariantDir('build/release', 'src')

# # We define our debug build environment in a similar fashion...
# debug_env = common_env.Clone()
# debug_env.Append(CPPDEFINES=['DEBUG'])
# debug_env.VariantDir('build/debug', 'src')

# # Now that all build environment have been defined, let's iterate over
# # them and invoke the lower level SConscript files.
# for mode, env in dict(release=release_env, 
#     	       	      debug=debug_env).items():
#     env.SConscript('build/%s/SConscript' % mode, {'env': env})
for mode, env in dict(release=release_env).items():
    env.SConscript('build/%s/SConscript' % mode, {'env': env})    


