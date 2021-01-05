import imageio

images = []
images2 = []
for e in range(100):
    img_name ='fake_result_epoch_{:03}.png'.format(e)
    img_name2 = 'real_result_epoch_{:03}.png'.format(e)
    images.append(imageio.imread(img_name))
    images2.append(imageio.imread(img_name2))
imageio.mimsave('generation_animation.gif', images, fps=2)
imageio.mimsave('real_animation.gif', images2, fps=2)