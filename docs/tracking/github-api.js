// GitHub API helper for auto-tracking bets
const GITHUB_CONFIG = {
  owner: 'pbwitt',
  repo: 'fourth-and-value',
  branch: 'main',
  filePath: 'docs/data/bets/bets.csv'
};

function getGitHubToken() {
  return localStorage.getItem('github_token');
}

function setGitHubToken(token) {
  localStorage.setItem('github_token', token);
  console.log('GitHub token saved! Auto-tracking enabled.');
}

async function autoTrackBet(betData) {
  const token = getGitHubToken();

  if (!token) {
    const csvRow = Object.values(betData).join(',');
    const setupInstructions = `GitHub token not set. To enable auto-tracking:

1. Create token: https://github.com/settings/tokens/new
2. Give it 'repo' permissions
3. Run in console: setGitHubToken('your_token_here')

Manual fallback - add to data/bets/bets.csv:

${csvRow}`;
    alert(setupInstructions);
    return false;
  }

  try {
    const { owner, repo, branch, filePath } = GITHUB_CONFIG;
    const apiUrl = `https://api.github.com/repos/${owner}/${repo}/contents/${filePath}`;

    const getResponse = await fetch(apiUrl, {
      headers: {
        'Authorization': `token ${token}`,
        'Accept': 'application/vnd.github.v3+json'
      }
    });

    if (!getResponse.ok) {
      throw new Error(`Failed to fetch file: ${getResponse.statusText}`);
    }

    const fileData = await getResponse.json();
    const currentContent = atob(fileData.content);
    const csvRow = Object.values(betData).join(',');
    const newContent = currentContent.trim() + '\n' + csvRow + '\n';

    const updateResponse = await fetch(apiUrl, {
      method: 'PUT',
      headers: {
        'Authorization': `token ${token}`,
        'Accept': 'application/vnd.github.v3+json',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        message: `Track bet: ${betData.player || 'Team Total'} ${betData.market_type} ${betData.side} ${betData.line}`,
        content: btoa(newContent),
        sha: fileData.sha,
        branch: branch
      })
    });

    if (!updateResponse.ok) {
      const errorData = await updateResponse.json();
      throw new Error(errorData.message || updateResponse.statusText);
    }

    alert(`Bet tracked!\n\n${betData.player || 'Team Total'} ${betData.market_type} ${betData.side} ${betData.line}\nStake: $${betData.stake_dollars}\n\nView at: https://fourthandvalue.com/tracking/`);
    return true;

  } catch (error) {
    console.error('Error tracking bet:', error);
    const csvRow = Object.values(betData).join(',');
    alert(`Error: ${error.message}\n\nManual fallback - add to data/bets/bets.csv:\n\n${csvRow}`);
    return false;
  }
}

window.autoTrackBet = autoTrackBet;
window.setGitHubToken = setGitHubToken;
