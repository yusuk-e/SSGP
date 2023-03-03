# Symplectic Spectrum Gaussian Processes | 2022
# Yusuke Tanaka

ns = [10,15,20,30,50]
timescales = [3,5,10]
sigmas = [0.1]
etas = [0,0.05]
val_rate = 0.3
commands = [ ]

#--pendulum--
names = ['pendulum']
names.each do |name|
  etas.each do |eta|
    cmd = "python generate_test.py --name #{name} --test_samples 25 --timescale 100 --T 15 --radius_a 1. --radius_b 1.3 --eta #{eta} --input_dim 2"
    commands.push cmd
    ns.each do |n|
      val_samples = [5, (n * val_rate).to_i].max
      n = n - val_samples
      sigmas.each do |sigma|
        timescales.each do |timescale|
          cmd = "python generate_train.py --name #{name} --train_samples #{n} --timescale #{timescale} --T 10 --radius_a 1. --radius_b 1.3 --sigma #{sigma} --val_samples #{val_samples} --eta #{eta} --input_dim 2"
          commands.push cmd
        end
      end
    end
  end
end

#--Duffing--
names = ['duffing']
names.each do |name|
  etas.each do |eta|
    cmd = "python generate_test.py --name #{name} --test_samples 25 --timescale 100 --T 15 --radius_a 1.5 --radius_b 0.5 --eta #{eta} --input_dim 2"
    commands.push cmd
    ns.each do |n|
      val_samples = [5, (n * val_rate).to_i].max
      n = n - val_samples
      sigmas.each do |sigma|
        timescales.each do |timescale|
          cmd = "python generate_train.py --name #{name} --train_samples #{n} --timescale #{timescale} --T 10 --radius_a 1.5 --radius_b 0.5 --sigma #{sigma} --val_samples #{val_samples} --eta #{eta} --input_dim 2"
          commands.push cmd
        end
      end
    end
  end
end

#--double pendulum--
names = ['d_pendulum']
names.each do |name|
  etas.each do |eta|
    cmd = "python 2d_generate_test.py --name #{name} --test_samples 25 --timescale 100 --T 15 --eta #{eta} --input_dim 4"
    commands.push cmd
    ns.each do |n|
      val_samples = [5, (n * val_rate).to_i].max
      n = n - val_samples
      sigmas.each do |sigma|
        timescales.each do |timescale|
          cmd = "python 2d_generate_train.py --name #{name} --train_samples #{n} --timescale #{timescale} --T 10 --sigma #{sigma} --val_samples #{val_samples} --eta #{eta} --input_dim 4"
          commands.push cmd
        end
      end
    end
  end
end

#--execute--
commands.each do |cmd|
  puts cmd
  system cmd
end
