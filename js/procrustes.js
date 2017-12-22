var ImageRater, createKDTree, numeric;

numeric = require('numeric');

createKDTree = require('static-kdtree');

ImageRater = class ImageRater {
  constructor() {
    this.procrustes = this.procrustes.bind(this);
    null;
  }

  mean(M) {
    var arr, means;
    return means = (function() {
      var i, len, results;
      results = [];
      for (i = 0, len = M.length; i < len; i++) {
        arr = M[i];
        results.push((arr.reduce(function(t, s) {
          return t + s;
        })) / arr.length);
      }
      return results;
    })();
  }

  scale(M) {
    var s;
    // there's a bug in this fnc, so norm2 returns the squared L2 norm
    s = Math.sqrt(numeric.norm2(M) / (M[0].length * 2));
    return {
      scaled: numeric.div(M, s),
      s: s
    };
  }

  rotateAlign(src, ref) {
    var N, den, i, ii, num, ref1, theta, w, x, y, z;
    // apply procruste's rotation (minimize SSD)
    N = src[0].length - 1;
    [x, y] = src;
    [w, z] = ref;
    num = den = 0;
    for (ii = i = 0, ref1 = N; i <= ref1; ii = i += 1) {
      num += w[ii] * y[ii] - z[ii] * x[ii];
      den += w[ii] * x[ii] + z[ii] * y[ii];
    }
    //console.log num
    theta = Math.atan(num / den);
    return theta;
  }

  procrustes(src, ref, nIters) {

    var N, add, cycle, den, dtheta, final, i, ii, j, mul, num, ref1, ref2, refMean, refS, refWide, rotMat, srcMean, srcS, srcT, theta, tree, w, x, y, z;
    if (true) {
      src = src.slice();
      ref = ref.slice();
    }
    refMean = this.mean(ref);
    srcMean = this.mean(src);
    src = [numeric.sub(src[0], srcMean[0]), numeric.sub(src[1], srcMean[1])];
    ref = [numeric.sub(ref[0], refMean[0]), numeric.sub(ref[1], refMean[1])];
    ({
      scaled: src,
      s: srcS
    } = this.scale(src));
    ({
      scaled: ref,
      s: refS
    } = this.scale(ref));
    srcT = numeric.transpose(src);
    refWide = numeric.transpose(ref);
    tree = createKDTree(refWide);
    N = src[0].length - 1;
    theta = 0;
    for (cycle = i = 0, ref1 = nIters; i <= ref1; cycle = i += 1) {
      // get new ref based on nearest neighbor
      num = den = 0;
      for (ii = j = 0, ref2 = N; j <= ref2; ii = j += 1) {
        x = src[0][ii];
        y = src[1][ii];
        [w, z] = refWide[tree.nn([x, y])];
        num += x * z - y * w;
        den += x * w + y * z;
      }
      dtheta = Math.atan(num / den);
      rotMat = [[Math.cos(dtheta), -Math.sin(dtheta)], [Math.sin(dtheta), Math.cos(dtheta)]];
      src = numeric.dot(rotMat, src);
      theta += dtheta;
      //{scaled:src, s: srcS} = @scale(src)
      //console.log(srcS)
      //console.log(cycle);
    }
    add = numeric.add;
    mul = numeric.mul;
    //src is 2 points
    final = [add(mul(src[0], refS), refMean[0]), add(mul(src[1], refS), refMean[1])];
    return {
      mean: refMean,
      scale: refS,
      rot: rotMat,
      src: src,
      final: final
    };
  }

  longToWide(arr) {
    var i, ii, ref1, results;
    results = [];
    for (ii = i = 0, ref1 = arr[0].length; i <= ref1; ii = i += 1) {
      results.push([arr[0][ii], arr[1][ii]]);
    }
    return results;
  }

};

module.exports = ImageRater;
