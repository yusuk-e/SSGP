# Symplectic Spectrum Gaussian Processes | 2022
# Yusuke Tanaka

names = ['pendulum','duffing','d_pendulum']
timescales = [3,5,10]
samples = [10,15,20,30,50]
sigmas = [0.1]
sets = (0..4).to_a
num_basis = [100,250,500]
total_steps = 10000
batch_time = 1

#--conservation--
eta = 0
commands = [ ]
num_basis.each do |basis|
  sigmas.each do |sigma|
    samples.each do |sample|
      timescales.each do |timescale|
        names.each do |name|
          sets.each do |set|
            cmd = "python train.py --name #{name} --s #{set} --total_steps #{total_steps} --sigma #{sigma} --eta #{eta} --sample #{sample} --timescale #{timescale} --batch_time #{batch_time} --num_basis #{basis}"
            commands.push cmd
            cmd = "python test.py --name #{name} --s #{set} --sigma #{sigma} --eta #{eta} --sample #{sample} --timescale #{timescale} --num_basis #{basis}"
            commands.push cmd
          end
        end
      end
    end
  end
end

#--dissipation--
etas = [0.05]
num_basis.each do |basis|
  sigmas.each do |sigma|
    etas.each do |eta|
      samples.each do |sample|
        timescales.each do |timescale|
          names.each do |name|
            sets.each do |set|
              cmd = "python train.py --name #{name} --s #{set} --total_steps #{total_steps} --sigma #{sigma} --eta #{eta} --sample #{sample} --timescale #{timescale} --batch_time #{batch_time} --num_basis #{basis} --friction"
              commands.push cmd
              cmd = "python test.py --name #{name} --s #{set} --sigma #{sigma} --eta #{eta} --sample #{sample} --timescale #{timescale} --num_basis #{basis} --friction"
              commands.push cmd
              cmd = "python test.py --name #{name} --s #{set} --sigma #{sigma} --eta #{eta} --sample #{sample} --timescale #{timescale} --num_basis #{basis} --friction --task2"
              commands.push cmd
            end
          end
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

