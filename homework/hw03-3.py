
from mrjob.job import MRJob
class MRWordFrequencyCount(MRJob):

    def mapper(self, _, line):
        cool = line.split(',')[7]
        if cool.isdigit():
            x=int(cool)
            if x > 0:
                yield "rating", int(line.split(',')[3])
                
    def reducer(self, key, values):
      i,totalL,totalW=0,0,0
      for i in values:
        totalL += 1
        totalW += i     
      yield "avg", totalW/float(totalL)
    

if __name__ == '__main__':
    MRWordFrequencyCount.run()
