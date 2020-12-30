import os
import shlex
import subprocess

def continous(source, out, start, vframes, fps):
    cmd = ('ffmpeg -y -start_number {0:d} -framerate {2:d} -i {1!s} -hide_banner -crf 5 '+ \
           '-preset slow -c:v libx264  -pix_fmt yuv420p')
    cmd = cmd.format(start, source, fps)
    if not vframes is None:
        cmd += ' -vframes {0:d}'.format(vframes)
    cmd += ' ' + out
    return cmd

def extend(source, out, dur, fps):
    cmd = ('ffmpeg -y -i {0!s} -vf tpad=stop_mode=clone:' + \
           'stop_duration={1:f} {2!s}').format(source, dur, out)
    return cmd

def concat(b, out, a, reverse = False):
    cmd = 'ffmpeg -y -f concat -safe 0 -i ' +\
        '<(for f in \'{0!s}\' \'{1!s}\'; do echo \"file \'$f\'\"; done) ' + \
        '-c copy {2!s}'
    if reverse:
        cmd = cmd.format(b, a, out)
    else:
        cmd = cmd.format(a, b, out)
    return cmd

def still(src, out, dur):
    cmd = 'ffmpeg -loop 1 -i {0!s} -c:v libx264 -t {1:f} -pix_fmt yuv420p {2!s}'
    cmd = cmd.format(src,dur, out)
    return cmd

def blank(src, out, dur, fps):
    cmd = 'ffmpeg -y -f lavfi -r {0:d} -i color=white:600x400:d={1:f} -pix_fmt yuv420p {2!s}'
    cmd = cmd.format(fps, dur, out)
    return cmd

def chain(cmds, args, source, out, suffix):
    out_p = out + '_' + suffix + '{0:d}.mp4'
    src = source
    cmd_chain = []
    to_remove = []
    for i,(cmd,arg) in enumerate(zip(cmds, args)):
        out_i = out_p.format(i)
        cmd_chain.append(cmd(src, out_i, *arg))
        to_remove.append(out_i)
        src = out_i

    cmd_chain.append('mv {0!s} {1!s}.mp4'.format(to_remove[-1], out))
    cmd_chain.append(('rm ' + ' '.join(to_remove[:-1])))
    return cmd_chain


def pause(source, vframes, fps, loc, dur, out):
    out1 = out + '_p1'
    cmd = chain([continous, extend],
                  [(1, loc, fps), (dur, fps)],
                  source, out1, 'a')
    cmd += chain([continous, concat],
                 [(loc, vframes, fps), (out1+'.mp4', False)],
                source, out, 'b')
    cmd.append('rm ' + out1+'.mp4')
    return cmd

def stimuli(a, b, fps, dur, out):
    d = os.path.dirname(__file__)
    src = os.path.join(d, 'white_600_400.png')
    cmds = chain([blank, concat, concat],
                 [(dur, fps), (a, False), (b, True)],
                 src, out, 'e')
    return cmds

def run_cmd(cmds):
    for cmd in cmds:
        print(cmd)
        subprocess.run(cmd, check=False, shell = True, executable='/bin/bash' )

def continous_movie(source, out, fps = 60, vframes = None):
    cmd = continous(source, out + '.mp4', 0, vframes, fps)
    run_cmd([cmd])

def paused_movie(source, out, fps = 60, loc = 20, dur = 0.5):
    cmd = pause(source, None, fps, loc, dur, out)
    run_cmd(cmd)

def stimuli_movie(a, b, out, fps = 60, dur = 0.5):
    cmds = stimuli(a, b, fps, dur, out)
    run_cmd(cmds)
